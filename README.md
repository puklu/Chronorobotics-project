# How to use the tool

The tool uploads maps from `~/.ros` to map_bucket. The corresponding metadata for the maps for an environment is uploaded to env_bucket in the db.

The tool can also be used to fetch maps from the db. Fetches the zipped map files from the db and then extracts them to `~/.ros` directory.

                
                 
1. cd into ```project/scripts/```.

2. To upload all the recorded maps so far for all the environments in one go, then if the pre-requisites are met,

        Pre-requisites:
            If you want to upload all the recorded maps so far, then maps  for each environment should be compressed and present in the directory ```objects/maps/``` , with map being separated into different directories based on the environment.
                    
            For example: all the maps for environment 'env0' should be in the directory ```objects/maps/env0/```
                            all the maps for environment 'env1' should be in the directory ```objects/maps/env1/```
                            and so on...
 
          
        Run:    
        python main.py

3. To upload a particular map for an environment from `~/.ros`,
            
            python main.py -e <environment-to-which-map-belongs> -u <map-to-upload> 

4. To fetch all the maps for an environment for the db,
            
            python main.py -e <environment-name>
            
5. To fetch a particular map for an environment,
           
            python main.py -e <environment-name> -m <map-name>

6. To fetch only environment details (for eg map metadata),
            
            python main.py -oe <environment-name> 
            
                       

Note: The fetched maps from db are stored in `~/.ros`
