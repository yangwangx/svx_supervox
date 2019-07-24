from distutils.core import setup, Extension
import shutil

module = Extension('EvalSPModule', sources = ['eval_superpixel.cpp'])

setup(name = 'PackageName', 
      version = '1.0',
      description = 'This is a Python wrapper for eval_superpixel.cpp',
      ext_modules = [module])

# copy file
# naming can be different for different environment
shutil.copy2('./build/lib.linux-x86_64-3.6/EvalSPModule.cpython-36m-x86_64-linux-gnu.so', 'EvalSPModule.so')
