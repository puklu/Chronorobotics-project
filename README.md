# How to use the tool

Pre-requisites:
    The maps for each environment should be present in the directory ```objects\maps\``` , with map being separated into different directories based on the environment.
            
    For example: all the maps for environment 'env0' should be in the directory ```objects\maps\env0\```
                 all the maps for environment 'env1' should be in the directory ```objects\maps\env1\```
                 and so on...
                 
                 
1. cd into ```project/scripts```.
2. Run from CLI:  `python main.py -m<env0> -e<env0>` , 
            where -m argument is the environment name for which all the maps are to be fetched,
                  -e argument is the environment name for which environment details are to be fetched.
                  Just the name of the environment in -m should be enough to fetch maps for an environment.
                
        Note: 1. Atleast one of the arguments should be provided!
              2. If only m is provided, both the environment and maps will be fetched, both being from the same environment.
              3. If only e is provided, only the environment will be fetched.
              4. If both are provided, both maps and environment will be fetched, where the maps can be from a different environment and the environment object can of an environment different from the maps' environment. 
                

3. The fetched map objects are stored in ```project/fetched_objects/maps``` directory as ```.pkl``` files.
   The fetched environment object is stored in ```project/fetched_objects/environment``` directory```.pkl``` files.
