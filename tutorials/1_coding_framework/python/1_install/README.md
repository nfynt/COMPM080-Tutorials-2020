# Python - Coding Framework

## Installation

1. download and install [anaconda with python 3](https://www.anaconda.com/distribution/#download-section)

2. Open Anaconda environment  
    Windows  
    [Anaconda Prompt]( https://docs.anaconda.com/anaconda/user-guide/faq/#installing-anaconda )  
    [Anaconda Prompt-getting-started]( https://docs.anaconda.com/anaconda/user-guide/getting-started/ )
    
    Linux/OSX  
    + option 1 [Manually init]( https://docs.anaconda.com/anaconda/user-guide/faq/#installing-anaconda )  
        * first run source <path to conda>/bin/activate and then run conda init.  
    + option 2 [Add to PATH]( https://askubuntu.com/questions/760311/anaconda-i-have-to-type-export-path-anaconda3-bin-path-everytime-i-rerun )
    
3. cd to this repository `tutorials/1_coding_framework/python`
4. create a new virtual environment geo_tutorials    
`conda create -n  geo_tutorials  python=3.6  pip  notebook  jupyter  matplotlib  `  

5. activate new environment
`conda activate geo_tutorials`  
reference:
    ````
    (base) >conda create -n  geo_tutorials  python=3.6  pip  matplotlib
    Solving environment: done


    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.4
      latest version: 4.8.0

    Please update conda by running

        $ conda update -n base conda



    ## Package Plan ##

      environment location: D:\anaconda\envs\geo_tutorials

      added / updated specs:
        - pip
        - python=3.6


    The following NEW packages will be INSTALLED:

        certifi:        2019.11.28-py36_0
        pip:            19.3.1-py36_0
        python:         3.6.9-h5500b2f_0
        ...
        ...

    Proceed ([y]/n)? y

    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    #
    # To activate this environment, use
    #
    #     $ conda activate geo_tutorials
    #
    # To deactivate an active environment, use
    #
    #     $ conda deactivate
    
    (base) >conda activate geo_tutorials
    
    (geo_tutorials) >
    
    ````

6. Verify python and pip (Linux/OSX)
    ````
    (base) $ which python3
    /home/xxx/anaconda3/bin/python3
    (base) $ which pip
    /home/xxx/anaconda3/bin/pip
    
    ````

7. install pyrender via `pip install pyrender`
   (pyrender-0.1.33)
   
    **If you’re on MacOS**, 
    you’ll need to pre-install pyrender's fork of pyglet, which enables OpenGL contexts on MacOS.
    ```
    git clone https://github.com/mmatl/pyglet.git
    cd pyglet
    pip install .
   ```
8. launch a hello world example    
    `python hello_world.py `  
    ![]( ../imgs/hello.jpg )
    
9. cd `/COMPM080-Tutorials-2020-private/tutorials/1_coding_framework/python`  
    lanuch jupyter notebook by 
    `jupyter notebook --port 12000 --NotebookApp.token=''`  
    
10. and open [hello_world.ipynb](hello_world.ipynb)  

11. if you encountered pyglet exceptions, it may due to jupyter notebook is not running in `geo_tutorials` environment  
     please install the following packages:  
     $`conda deactivate`  
     $`conda install nb_conda_kernels`  
     $`conda install -n geo_tutorials ipykernels`  
     $`conda activate geo_tutorials`  
     $`jupyter notebook --port 12000 --NotebookApp.token=''`  
     $ open `hello_world.ipynb`  
     $ switch environment  
     ![](../imgs/jupyter.png)


* option reading:  
[pyrender install document](https://pyrender.readthedocs.io/en/latest/install/index.html)  












