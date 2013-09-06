
//@HEADER
// ************************************************************************
//
//               HPCG: Simple Conjugate Gradient Benchmark Code
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER

// Changelog
//
// Version 0.3
// - Added timing of setup time for sparse MV
// - Corrected percentages reported for sparse MV with overhead
//
/////////////////////////////////////////////////////////////////////////

// Main routine of a program that calls the HPCG conjugate gradient
// solver to solve the problem, and then prints results.

#include <fstream>
#include <iostream>
#ifdef DEBUG
using std::cin;
#endif
using std::endl;

#include <cstdlib>
#include <vector>

#include "hpcg.hpp"

#ifndef HPCG_NOMPI
#include <mpi.h> // If this routine is not compiled with HPCG_NOMPI
#endif

#ifndef HPCG_NOOPENMP
#include <omp.h> // If this routine is not compiled with HPCG_NOOPENMP
#endif

#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "SetupHalo.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "spmvref.hpp"
#include "symgsref.hpp"
#include "ComputeResidual.hpp"
#include "CG.hpp"
#include "CGref.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "CGData.hpp"

int main(int argc, char *argv[]) {
    
    
#ifndef HPCG_NOMPI
    
    MPI_Init(&argc, &argv);

    int size, rank; // Number of MPI processes, My process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    
    int size = 1; // Serial case (not using MPI)
    int rank = 0;
    
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, &params);

  int numThreads = 1;

#ifndef HPCG_NOOPENMP
#pragma omp parallel
  numThreads = omp_get_num_threads();
#endif

#ifdef DEBUG
    if (size < 100) HPCG_fout << "Process "<<rank<<" of "<<size<<" is alive with " << numThreads << " threads." <<endl;
#endif
    
#ifdef DEBUG
    if (rank==0)
    {
        int junk = 0;
        HPCG_fout << "Press enter to continue"<< endl;
        cin >> junk;
    }
#ifndef HPCG_NOMPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
    
    
#ifdef NO_PRECONDITIONER
    bool doPreconditioning = false;
#else
    bool doPreconditioning = true;
#endif
    
    local_int_t nx,ny,nz;
    nx = (local_int_t)params.nx;
    ny = (local_int_t)params.ny;
    nz = (local_int_t)params.nz;

#ifdef HPCG_DEBUG
    double t1 = mytimer();
#endif

    Geometry geom;
    GenerateGeometry(size, rank, numThreads, nx, ny, nz, geom);

    SparseMatrix A;
    CGData data;
    double *x, *b, *xexact;
    GenerateProblem(geom, A, &x, &b, &xexact);
    SetupHalo(geom, A);
    initializeCGData(A, data);

#ifdef HPCG_DEBUG
    if (rank==0) HPCG_fout << "Total setup time (sec) = " << mytimer() - t1 << endl;
#endif


    //if (geom.size==1) WriteProblem(A, x, b, xexact);

    // Use this array for collecting timing information
    std::vector< double > times(9,0.0);

    // Call user-tunable set up function.
    double t7 = mytimer(); OptimizeProblem(geom, A, data, x, b, xexact); t7 = mytimer() - t7;
    times[7] = t7;
    
    // Call Reference SPMV and SYMGS. Compute Optimization time as ratio of times in these routines

    local_int_t nrow = A.localNumberOfRows;
	local_int_t ncol = A.localNumberOfColumns;

	double * x_overlap = new double [ncol]; // Overlapped copy of x vector
	double * b_computed = new double [nrow]; // Computed RHS vector

	// Test symmetry of matrix

	// First load vector with random values
	for (int i=0; i<nrow; ++i) {
		x_overlap[i] = ((double) rand() / (RAND_MAX)) + 1;
	}

	int ierr = 0;
	int numberOfCalls = 10;
	double t_begin = mytimer();
	for (int i=0; i< numberOfCalls; ++i) {
#ifndef HPCG_NOMPI
		ExchangeHalo(A,x_overlap);
#endif
		ierr = spmvref(A, x_overlap, b_computed); // b_computed = A*x_overlap
		if (ierr) HPCG_fout << "Error in call to spmv: " << ierr << ".\n" << endl;
		ierr = symgsref(A, x_overlap, b_computed); // b_computed = Minv*y_overlap
		if (ierr) HPCG_fout << "Error in call to symgs: " << ierr << ".\n" << endl;
	}
    times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.

    int global_failure = 0; // assume all is well: no failures

    int niters = 0;
    int totalNiters = 0;
    double normr = 0.0;
    double normr0 = 0.0;
    int maxIters = 50;

    /* Compute the residual reduction for the natural ordering and reference kernels. */
    std::vector< double > ref_times(9,0.0);
    double tolerance = 0.0; // Set tolerance to zero to make all runs do max_iter iterations
    int err_count = 0;
    for (int i=0; i< numberOfCalls; ++i) {
    	for (int j=0; j< A.localNumberOfRows; ++j) x[j] = 0.0; // start x at all zeros
    	ierr = CGref( geom, A, data, b, x, maxIters, tolerance, niters, normr, normr0, &ref_times[0], true);
    	if (ierr) ++err_count; // count the number of errors in CG
	totalNiters += niters;
    }
    if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to reference CG." << endl;
    double ref_tolerance = normr / normr0;
    int ref_iters = niters;
    
    totalNiters = 0;
    niters = 0;
    normr = 0.0;
    normr0 = 0.0;
    err_count = 0;
    int tolerance_failures = 0;

    int opt_maxIters = 10*maxIters;
    int opt_iters = 0;
    double opt_worst_time = 0.0;

    /* Compute the residual reduction and residual count for the user ordering and optimized kernels. */
    for (int i=0; i< numberOfCalls; ++i) {
    	for (int j=0; j< A.localNumberOfRows; ++j) x[j] = 0.0; // start x at all zeros
        double last_cummulative_time = times[0];
    	ierr = CG( geom, A, data, b, x, opt_maxIters, ref_tolerance, niters, normr, normr0, &times[0], true);
    	if (ierr) ++err_count; // count the number of errors in CG
        if (normr / normr0 > ref_tolerance) ++tolerance_failures; // the number of failures to reduce residual

        // pick the largest number of iterations to guarantee convergence
        if (niters > opt_iters) opt_iters = niters;

        double current_time = times[0] - last_cummulative_time;
        if (current_time > opt_worst_time) opt_worst_time = current_time;

	totalNiters += niters;
    }
    if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to optimized CG." << endl;
    if (tolerance_failures) {
      global_failure = 1;
      if (rank == 0)
        HPCG_fout << "Failed to reduce the residual " << tolerance_failures << " times." << endl;
    }

    double total_runtime = 60; // run for at least one minute
    numberOfCalls = int(total_runtime / opt_worst_time);
    if (numberOfCalls < 1) numberOfCalls = 1; // run CG at least once

    for (int i=0; i< numberOfCalls; ++i) {
    	for (int j=0; j< A.localNumberOfRows; ++j) x[j] = 0.0; // Zero out x
    	ierr = CG( geom, A, data, b, x, maxIters, tolerance, niters, normr, normr0, &times[0], doPreconditioning);
    	if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
    	if (rank==0) HPCG_fout << "Call [" << i << "] Scaled Residual [" << normr/normr0 << "]" << endl;
	totalNiters += niters;
    }
    
    // Compute difference between known exact solution and computed solution
    // All processors are needed here.
#ifdef DEBUG
    double residual = 0;
    ierr = ComputeResidual(A.localNumberOfRows, x, xexact, &residual);
    if (ierr) HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
    if (rank==0)
    HPCG_fout << "Difference between computed and exact  = " << residual << ".\n" << endl;
#endif

    // Report results to YAML file
    ReportResults(geom, A, totalNiters, normr/normr0, &times[0]);

    // Clean up
    destroyMatrix(A);
    destroyCGData(data);
    delete [] x;
    delete [] b;
    delete [] xexact;
    
    HPCG_Finalize();

    // Finish up
#ifndef HPCG_NOMPI
    MPI_Finalize();
#endif
    return 0 ;
} 