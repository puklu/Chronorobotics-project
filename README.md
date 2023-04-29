# How to use the tool

The tool uploads maps to map_bucket. The corresponding metadata for the maps for an environment is uploaded to env_bucket in the db.

The tool can also be used to fetch maps from the db. Fetches the zipped map files from the db and then extracts them to `~/.ros` directory.

                
                 
1. cd into ```project/scripts/Map_management/```.


2. To upload a particular map for an environment from `~/.ros`,
            
            python main.py -e <environment-to-which-map-belongs> -u <map-to-upload> -snode <starting-node> -enode <ending-node>
            
3. To fetch a particular map for an environment,
           
            python main.py -e <environment-name> -m <map-name>            

4. To fetch all the maps for an environment for the db,
            
            python main.py -e <environment-name>
            
5. To fetch only environment details (for eg map metadata),
            
            python main.py -oe <environment-name> 
            
6. To find an optimum path between two nodes,
            
            python main.py -e <environment-name> -findpath <first-node> <second-node>
            
7. To delete a specific map from an environment,
            
            python main.py -e <environment-name> -delamap <name-of-the-map>

8. To delete all the maps for a particular environment,

            python main.py -delmaps <name-of-the-environment>
                        
            
9. To upload all the recorded maps for all the environments in one go,

        Pre-requisites:
            If you want to upload all the recorded maps so far, then maps  for each environment should be present in the directory ```objects/maps/``` , with map being separated into different directories based on the environment.
                    
            For example: all the maps for environment 'env0' should be in the directory ```objects/maps/env0/```
                            all the maps for environment 'env1' should be in the directory ```objects/maps/env1/```
                            and so on...
 
        Note: 1. starting-node and ending-node for each map has to be manually entered in the code !
              2. MANIPULATE flag should be set to False by default. 
          
        Run:    
        python main.py        
                       

Note: The fetched maps from db are stored in `~/.ros`


( 
While uploading maps, in case the user wants to manually manipulate the distance and heuristic cost between the nodes for testing,

    python main.py -e <environment-to-which-map-belongs> -u <map-to-upload> -snode <starting-node> -enode <ending-node> -mani <manipulated-distance> <manipulated-cost>
            
)
