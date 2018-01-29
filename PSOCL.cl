typedef struct particle {
    int dimensionnum;
    double *present, *pbest, fitness,*v, w;
} particle;

typedef struct clswarm {
    int partnum, dimnum;
    double *gbest, w;
    particle *part;
} clswarm;


//I am going to try to add everything in one function, there is 2 dimensions, one for the swarm particles
//and one for the particle dimensions, using the 2d array for get_global_id, i should be able to
//add everything that way.
__kernel void PSO_vector_add(particle *part, double  *rand, clswarm school){
    int gid=get_global_id(1);
    part.v[gid]= school.w*part.v[gid] + rand[0]*(part.pbest[gid]-part.v[gid]) + rand[1]*(gbest[gid]-part.v[gid]);
    part.present[gid]=part.v[gid]+part.present[gid];
}