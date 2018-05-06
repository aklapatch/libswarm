///cnstrctrex.cpp
/** testing constructor behavior */
/** conclusion: Object=constructor is a valid way to re-initialize data */

#include <iostream>

class Test {
    private:
        int tst1;

    public:
        Test() {
            tst1=0;
        }

        Test(int in){
            tst1=in;
        }

        ~Test() {}

        int getInt() {
            return tst1;
        }
        void setInt(int in){
            tst1=in;
        }
};

int main(){

    Test other;
    other=Test(13);

    std::cout << "other= " << other.getInt() << "\n";

    return 0;
}