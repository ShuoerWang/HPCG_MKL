
import os
import sys
import subprocess
import shutil

#compiler: 0-gcc, 1=clang, 2=ic ol: optimization level
def change_compile_options(file_path, ol, compiler, target_file):
    with open(file_path, "r") as f:
        lines = f.readlines()

    search_str = "add_compile_options"
    new_options = ""
    if compiler == 0:
        new_options = "add_compile_options(" + "-fopt-info-optimized " + "-O" + str(ol) + ")\n"
    elif compiler == 1:
        new_options = "add_compile_options(" + "-Rpass " + "-O" + str(ol) + ")\n"
    else:
        new_options = "add_compile_options(" + "-qopt-report-file=stdout " + "-O" + str(ol) + ")\n"
 

    with open(file_path, "w") as f:
        for line in lines:
            if search_str in line:
                line = ""
            f.write(line)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line)
            if "cmake_minimum_required" in line:
                f.write(new_options)
            
        

if __name__ == '__main__':
    file_path = "../CMakeLists.txt"
    target_file = ""
    if len(sys.argv) > 1:
        target_file = sys.argv[1]

    for ol in range(0, 4):
        for compiler in range(0, 3):
            change_compile_options(file_path, ol, compiler, target_file)
            p_path = os.getcwd()
            path = os.getcwd()
            if compiler == 0:
                subprocess.run(['cmake','--fresh', '-D', 'CMAKE_CXX_COMPILER=g++', '..'])
                file_name = "g++_"+"O" + str(ol) + "/g++_"+"O"+ str(ol) +".txt"
                path += "/g++_"+"O" + str(ol)
                
            elif compiler == 1:
                subprocess.run(['cmake', '--fresh', '-D', 'CMAKE_CXX_COMPILER=clang++', '..'])
                file_name = "clang++_"+"O"+str(ol) + "/clang++_"+"O"+str(ol)+".txt"
                path += "/clang++_"+"O" + str(ol)

            elif compiler == 2:
                subprocess.run(['cmake', '--fresh', '-D', 'CMAKE_CXX_COMPILER=icpx', '..'])
                file_name = "icpx_"+"O"+ str(ol) + "/icpx_"+"O"+ str(ol) +".txt"
                path += "/icpx_"+"O" + str(ol)
            
            subprocess.run(['make', 'clean'])
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)
            with open(file_name, 'w') as f:
                f.write(subprocess.run(['make'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode())
            os.chdir(path)
            subprocess.run(['../xhpcg'])
            os.chdir(p_path)


