typedef struct particle {
    int dimensionnum;
    double *present, *pbest, fitness,*v, w;
} particle;

typedef struct clswarm {
    int partnum, dimnum;
    double *gbest, w;
    particle *part;
} clswarm;

__kernel void swarm_add(clswarm  school){
    int gid = get_global_id(0);
    PSO_vector_add(school.part[gid]);        
}

__kernel void PSO_vector_add(particle *part, double  *rand, double *gbest,double w){
    int gid=get_global_id(0);
    part.v[gid]= w*part.v[gid] + rand[0]*(part.pbest[gid]-part.v[gid]) + rand[1]*(gbest[gid]-part.v[gid]);
    part.present[gid]=part.v[gid]+part.present[gid];
}