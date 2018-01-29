/*PSOCL.cl
Houses the main function to add the particle positions.
*/

typedef struct particle {   //particle structure
    double *present, *pbest, fitness,*v;
} particle;

typedef struct clswarm {    //swarm structure
    int partnum, dimnum;
    double *gbest, w;
    particle *part;
} clswarm;

//I am going to try to add everything in one function, there is 2 dimensions, one for the swarm particles
//and one for the particle dimensions, using the 2d array for get_global_id, i should be able to
//add everything that way.
__kernel void PSO_vector_add(__local double  *rand, __global clswarm school){
    int dimenid=get_global_id(1);
    int partid=get_global_id(0);

    if(dimenid<dimnum&&partid<partnum) {

        school.part[partid].v[dimenid]=school.w*school.part[partid].v[dimenid]
        + rand[0]*(school.part[partid].pbest[dimenid]-school.part[partid].present[gid])
        + rand[1]*(school.gbest[dimenid]-school.part[partid].present[gid]);

        school.part[partid].present[dimenid]=school.part[partid].present[dimenid]+
        school.part[partid].v[dimenid];
    }    
}