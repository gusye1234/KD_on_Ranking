/*
<%
cfg['compiler_args'] = ['-std=c++11', '-undefined dynamic_lookup']
%>
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

ui randint(ui start, ui end)
{
    return (rand() % (end - start)) + start;
}

int randint_(int end)
{
    return rand()%end;
}

std::vector<int *> try_list(std::vector<int *> ok)
{
    ok.push_back(ok.back());
    return ok;
}

py::array_t<int> create_numpy(int shape1, int shape2, int init){
// using request() to obtain memory info
// using *.ptr to localize
// 
//     
    auto result = py::array_t<int>({shape1, shape2});
    py::buffer_info buf_result = result.request();
    int *ptr = (int*)buf_result.ptr;
    for(int i=0;i<buf_result.size;i++)
        ptr[i] = init;
    return result;
}


// std::vector<std::vector<ui>> sample_negative(int user_num, int item_num,int train_num, std::vector<std::vector<ui>> allPos, int neg_num){
py::array_t<int> sample_negative(int user_num, int item_num, int train_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int perUserNum = (train_num/user_num);
    int row = neg_num + 2;
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int*)buf_S.ptr;

    for(int user = 0; user < user_num; user++){
        std::vector<int> pos_item = allPos[user];
        // TODO
        // sort(pos_item.begin(), pos_item.end());

        for(int pair_i = 0;pair_i < perUserNum; pair_i++){
            int negitem = 0;
            ptr[(user*perUserNum+pair_i)*row] = user;
            ptr[(user * perUserNum + pair_i) * row + 1] = pos_item[randint_(pos_item.size())];
            for(int index=2; index < neg_num+2;index++){
                do{
                    negitem = randint_(item_num);
                }
                while(
                    find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end()
                );
                ptr[(user * perUserNum + pair_i) * row + index] = negitem;
            }
            // std::cout << "From Cpp: " << user << ' ' << ptr[(user * perUserNum + pair_i) * row + 1];
            // for(int i=0;i<neg_num;i++){
            //     std::cout << ptr[(user * perUserNum + pair_i) * row + i + 2]  << ' ';
            // }
            // std::cout<<std::endl;
        }
    }
    // std::cout<<(user_num*perUserNum)
    return S_array;
}

// py::array_t<int> sample_negative(int user_num, int item_num, int train_num, std::vector<std::vector<int>> allPos, int neg_num)
// {
//     int perUserNum = (train_num / user_num);
//     int row = neg_num + 2;
//     py::array_t<int> S_array = py::array_t<int>({train_num, row});
//     py::buffer_info buf_S = S_array.request();
//     std::cout << buf_S.size << std::endl;
//     int *ptr = (int *)buf_S.ptr;

//     for (int user = 0; user < user_num; user++)
//     {
//         std::vector<int> pos_item = allPos[user];
//         // TODO
//         // sort(pos_item.begin(), pos_item.end());

//         for (int pair_i = 0; pair_i < perUserNum; pair_i++)
//         {
//             int negitem = 0;
//             ptr[(user * perUserNum + pair_i) * row] = user;
//             ptr[(user * perUserNum + pair_i) * row + 1] = pos_item[randint_(pos_item.size())];
//             for (int index = 2; index < neg_num + 2; index++)
//             {
//                 do
//                 {
//                     negitem = randint_(item_num);
//                 } while (
//                     find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
//                 ptr[(user * perUserNum + pair_i) * row + index] = negitem;
//             }
//             // std::cout << "From Cpp: " << user << ' ' << ptr[(user * perUserNum + pair_i) * row + 1];
//             // for(int i=0;i<neg_num;i++){
//             //     std::cout << ptr[(user * perUserNum + pair_i) * row + i + 2]  << ' ';
//             // }
//             // std::cout<<std::endl;
//         }
//     }
//     return S_array;
// }

using namespace py::literals;

PYBIND11_MODULE(sampling, m)
{
    // srand(time(0));
    srand(2020);
    m.doc() = "example plugin";
    m.def("try_list", &try_list, "None", "ok"_a);
    m.def("randint", &randint_, "generate [0 end]", "end"_a);
    m.def("create_numpy", &create_numpy, "create empty numpy ", "shape1"_a, "shape2"_a, "init"_a=0);
    m.def("sample_negative", &sample_negative, "sampling negatives for batch", 
          "user_num"_a, "item_num"_a, "train_num"_a,"allPos"_a, "neg_num"_a);
}