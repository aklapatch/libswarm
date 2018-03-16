///distribute.cl
/** this kernel distributes the particles positions
 *  in gpu memory
 */

#define id(x) (get_global_id(x))

int get_global_id(int);

__kernel void distribute(__global float * lowerbound, 
                         __global float * upperbound,
                         __global float * gbest,
                         __global float * delta,
                         __global float * presents,
                         __global float * pbests,
                         __global float * v,
                         int partnum){
    ///id(1) is dimension number, id(0) is particle number
    delta[id(1)]=(upperbound[id(1)]-lowerbound[id(1)])/(partnum-1);
    ///set initial gbest position to 0
    gbest[id(1)]=0;

    ///distribute the particle between the upper and lower boundaries linearly
    presents[id(0)*partnum +id(1)]=id(0)*delta[id(1)] + lowerbound[id(1)];
    pbests[id(0)*partnum+id(1)]=0;
    v[id(0)*partnum + id(1)]=0;
}

