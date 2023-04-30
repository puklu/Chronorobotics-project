# Map Management System for Visual Teach and Repeat

## a) Efficient management of maps
MinIO db is used as the database service. Once the service is up and running, the tool uploads maps to `map` bucket. The corresponding metadata for the maps for an environment is uploaded to `env` bucket in the db.
1. cd into ```project/scripts/Map_management/```.


2. To upload a particular map for an environment from `~/.ros`,

            python main.py -e <environment-name> -u <map-name> -snode <starting-node> -enode <end_node>

The tool can fetch maps from the db. Fetches the zipped map files from the db and then extracts them to `~/.ros/fetched_maps` directory.

3. To fetch a particular map for an environment,
           
            python main.py -e <environment-name> -m <map-name>            

4. To fetch all the maps for an environment for the db,
            
            python main.py -e <environment-name>
            
5. To fetch only environment details (for eg map metadata). The details are saved as json in `project/results/` directory.
            
            python main.py -oe <environment-name> 
 
Maps can also be deleted from the db.           
            
6. To delete a specific map from an environment,
            
            python main.py -e <environment-name> -delamap <name-of-the-map>

7. To delete all the maps for a particular environment,

            python main.py -delmaps <name-of-the-environment>
 
## b) Finding an optimum sequence of maps 

Using the results of spectral anaylsis of the data in the db and metadata stored in `env` bucket,
the mathematical model of the environment is used to plan a path between any two nodes of the environment optimising for a number of criteria.


1. cd into ```project/scripts/Map_management/```.


2. To find an optimum path between two nodes,

            python main.py -e <environment-name> -findpath <first-node> <second-node>

The maps corresponding to the optimum path are fetched from the db into `~/.ros/fetched_maps/`

