#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h> // MPI: 包含 MPI 头文件

using namespace std;
using namespace chrono;

// MPI 编译指令示例:
// mpic++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o main


enum DataType { BOOL_DATA = 1, STRING_DATA = 2};
void send_vector_of_strings(const std::vector<std::string>& vec, int dest_rank, MPI_Comm comm) {
    // --- 第一阶段：发送元数据 ---

    // 1. 发送 vector 中的字符串数量
    size_t num_strings = vec.size();
    MPI_Send(&num_strings, 1, MPI_UNSIGNED_LONG, dest_rank, 0, comm);//发送给dest_rank号进程

    // 2. 准备并发送每个字符串的长度
    std::vector<size_t> lengths;
    lengths.reserve(num_strings);
    size_t total_chars = 0;
    for (const auto& s : vec) {
        lengths.push_back(s.length());
        total_chars += s.length();
    }
    MPI_Send(lengths.data(), num_strings, MPI_UNSIGNED_LONG, dest_rank, 1, comm);

    // --- 第二阶段：打包并发送实际数据 ---
    
    // 3. 将所有字符串打包到一个连续的 char 缓冲区
    std::string packed_data;
    packed_data.reserve(total_chars);
    for (const auto& s : vec) {
        packed_data.append(s);
    }

    // 4. 发送打包好的数据
    MPI_Send(packed_data.c_str(), packed_data.length(), MPI_CHAR, dest_rank, 2, comm);
}

std::vector<std::string> receive_vector_of_strings(int source_rank, MPI_Comm comm) {
    std::vector<std::string> vec;
    MPI_Status status;

    // --- 第一阶段：接收元数据 ---

    // 1. 接收字符串数量
    size_t num_strings;
    MPI_Recv(&num_strings, 1, MPI_UNSIGNED_LONG, source_rank, 0, comm, &status);

    // 2. 接收每个字符串的长度
    std::vector<size_t> lengths(num_strings);
    MPI_Recv(lengths.data(), num_strings, MPI_UNSIGNED_LONG, source_rank, 1, comm, &status);

    // --- 第二阶段：接收并解包实际数据 ---

    // 3. 计算总字符数并准备接收缓冲区
    size_t total_chars = std::accumulate(lengths.begin(), lengths.end(), (size_t)0);
    std::vector<char> packed_buffer(total_chars);

    // 4. 接收打包好的数据
    MPI_Recv(packed_buffer.data(), total_chars, MPI_CHAR, source_rank, 2, comm, &status);

    // 5. 解包数据，重建 vector<string>
    size_t current_pos = 0;
    for (size_t len : lengths) {
        // 从缓冲区的正确位置和长度构造 string
        vec.emplace_back(packed_buffer.data() + current_pos, len);
        current_pos += len;
    }

    return vec;
}

int main() // MPI: main 函数需要接收 argc 和 argv
{
    // MPI: 初始化 MPI 环境
	MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //获取全局进程组
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int rank_to_exclude = size - 1;
    //创建不包括size-1号进程的子进程组
    MPI_Group new_group;
    MPI_Group_excl(world_group, 1, &rank_to_exclude, &new_group);
    //根据新组创建新通信域
    MPI_Comm new_comm;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

	double time_hash = 0;
	double time_guess = 0;
	double time_train = 0;
	PriorityQueue q;

    // --- 模型训练 ---
    // 所有进程都加载和训练自己的模型副本。
	if (rank == 0) { // 只让主进程计时
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    } else {
        if(rank!=size-1)
        {
            // 其他除用于hash的进程也必须加载模型，但不需要计时
            q.m.train("/guessdata/Rockyou-singleLined-full.txt");
            q.m.order();
        }
    }
    if(new_comm!=MPI_COMM_NULL)
    {
        // MPI: 使用 MPI_Barrier 确保所有进程都完成了模型训练才继续
        MPI_Barrier(new_comm);
    }
    


    // --- 优先队列初始化 ---
    // 只有主进程(rank 0)负责初始化优先队列。
	if (rank == 0) {
        q.init();
        cout << "here" << endl;
    }

	int curr_num = 0;
    // MPI: 计时和 history 只在主进程中有意义
    auto start = (rank == 0) ? system_clock::now() : system_clock::time_point{};
	int history = 0;
    bool continue_looping = true;
    
    // --- 主循环 ---
    // 循环条件由主进程决定，并广播给其他进程
    if(rank!=size-1)
    {
        do
        {
            // 所有进程都调用 PopNext，它内部有并行的分发、计算、收集逻辑。

            q.PopNext(new_comm);
            
            if (rank == 0) {
                
                
                q.total_guesses = q.guesses.size();
                
                if (q.total_guesses - curr_num >= 100000)
                {
                    cout << "Guesses generated: " << history + q.total_guesses << endl;
                    curr_num = q.total_guesses;

                    
                    int generate_n = 10000000;
                    if (history + q.total_guesses > generate_n)
                    {
                        auto end = system_clock::now();
                        auto duration = duration_cast<microseconds>(end - start);
                        time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                        cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                        //cout << "Hash time:" << time_hash << "seconds" << endl;
                        cout << "Train time:" << time_train << "seconds" << endl;
                        continue_looping = false; // 标记循环应该结束
                    }
                }

                
                if (curr_num > 1000000)
                {
                    int data_type_id = STRING_DATA;
                    MPI_Send(&data_type_id, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD);
                    //0号进程将q.guesses分发给一个进程并维持循环条件
                    send_vector_of_strings(q.guesses,size-1,MPI_COMM_WORLD);
                    history += curr_num;
                    curr_num = 0;
                    q.guesses.clear();
                }
                
                if (q.priority.empty()) {
                    continue_looping = false;
                }
            }
            MPI_Bcast(&continue_looping, 1, MPI_CXX_BOOL, 0, new_comm);
           
            if(continue_looping==false)
            {
                int data_type_id = BOOL_DATA;
                MPI_Send(&data_type_id, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD); // 1. 发送标识
                MPI_Send(&continue_looping, 1, MPI_CXX_BOOL, size - 1, 0, MPI_COMM_WORLD); // 2. 发送数据
            }
        
        } while (continue_looping);
        
    }
    auto end0= (rank == 0) ? system_clock::now() : system_clock::time_point{};


    

    if(rank==size-1)
    {
        
        vector<string> q_guesses;
        bool continue_next=true;
        long long end1_microseconds;
        while(continue_next||!(q_guesses.empty()))
        {
            
            int flag = 0; // 标志位，表示是否有消息
            MPI_Status status;
            MPI_Iprobe(0, 0, MPI_COMM_WORLD, &flag, &status);      
            if(flag)
            {
                int data_type_id;
                MPI_Recv(&data_type_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                switch(data_type_id){
                    case STRING_DATA:{
                        std::vector<std::string> received_guesses = receive_vector_of_strings(0, MPI_COMM_WORLD);
                        q_guesses.insert(q_guesses.end(), received_guesses.begin(), received_guesses.end());
                        break;
                    }
                    case BOOL_DATA:{
                        MPI_Recv(
                            &continue_next,       // 接收缓冲区的地址
                            1,                     // 接收的数据项数量
                            MPI_CXX_BOOL,          // MPI数据类型，专门用于C++的bool
                            0,           // 消息来源进程的秩
                            0,           // 消息标签
                            MPI_COMM_WORLD,        // 通信域
                            MPI_STATUS_IGNORE      // 忽略返回的状态信息
                            );
                            break;
                    }
                }
                
            }
            auto start_hash = system_clock::now();
            bit32 batch_states[4][4];
            size_t total = q_guesses.size();
            for (size_t i = 0; i < total; i += 4) {
                std::string batch[4];
                size_t remain = total - i;
                size_t batch_size = (remain >= 4) ? 4 : remain;
                for (size_t j = 0; j < batch_size; ++j) {
                        batch[j] = q_guesses[i + j];
                        }
                    for (size_t j = batch_size; j < 4; ++j) {
                        batch[j] = "";
                        }
                MD5Hash(batch, batch_states);
            }
            q_guesses.clear();
            auto end_hash = system_clock::now();
            auto end1 = system_clock::now();
            end1_microseconds = time_point_cast<microseconds>(end1).time_since_epoch().count();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
        }
        cout << "Hash time:" << time_hash << "seconds" << endl;
        
        MPI_Send(
    &end1_microseconds,  // 发送数据的起始地址
    1,                   // 发送的数据项数量
    MPI_LONG_LONG,       // 发送数据的 MPI 数据类型
    0,                   // 目的地进程的秩 (主控进程是 0)
    100,        // 消息标签，用于区分不同类型的消息
    MPI_COMM_WORLD       // 通信域
);
    }
    if(rank==0)
    {
        MPI_Status status;
        long long received_end1_microseconds = 0;
        MPI_Recv(
    &received_end1_microseconds, // 接收数据缓冲区的起始地址
    1,                           // 接收的数据项数量
    MPI_LONG_LONG,               // 接收数据的 MPI 数据类型
    size - 1,                    // 消息来源进程的秩 (哈希进程)
    100,                // 消息标签，必须和发送方匹配
    MPI_COMM_WORLD,              // 通信域
    &status                      // 接收状态信息
    );
    long long end0_microseconds = time_point_cast<microseconds>(end0)
                                  .time_since_epoch()
                                  .count();
    long long start_microseconds = time_point_cast<microseconds>(start)
                                  .time_since_epoch()
                                  .count();
    long long tend=end0_microseconds>received_end1_microseconds?end0_microseconds:received_end1_microseconds;
    long long total_time=tend-start_microseconds;
    double total_times=total_time/1000000.0;
    cout<<"total time:"<<total_times<<"seconds"<<endl;
    }
    MPI_Comm_free(&new_comm);
    // 释放组资源
    MPI_Group_free(&new_group);
    MPI_Group_free(&world_group);
    // MPI: 释放 MPI 资源
	MPI_Finalize();
	return 0;
}