///distribute.cl
/** this kernel distributes the particles positions
 *  in gpu memory
 */

#define id(x) (get_global_id(x))

int get_global_id(int);

__kernel void distribute(__global float * lowerbound, 
                         __global float * upperbound,
                         __global float * delta,
                         __global float * presents,
                         __global float * pbests,
                         __global int * partnum){
    unsigned int dex[3] = {id(1), id(0)*(*partnum) +id(1), id(0)};

    ///id(1) is dimension number, id(0) is particle number
    delta[dex[0]]=(upperbound[dex[0]]-lowerbound[dex[0]])/(*partnum);

    ///distribute the particle between the upper and lower boundaries linearly
    presents[dex[1]]=dex[2]*delta[dex[0]] + lowerbound[dex[0]];
    pbests[dex[1]]=0;
}

