typedef struct particle {
    int dimensionnum;
    double *present, *pbest, fitness,*v, w;
} particle;

typedef struct clswarm {
    int partnum, dimnum;
    double *gbest, w;
    particle *part;
} clswarm;

__kernel void PSOadd(clswarm  school, double rand1, double rand2){
    int i, gid = get_global_id(0);
    for(i=0,i<school.dimnum,++i){
        school.part[gid].v[i]=school.w*school.part[gid].v[i] + 1.492*rand1*(gbest[i]-present[i])
        +1.492*rand2*(school.part[gid].pbest[i];
        school.part[gid].present[i]= school.part[gid].v[i] + school.part[gid].present[i];
    }    
        
    
}
