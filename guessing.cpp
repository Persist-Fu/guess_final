#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <numeric>
#include <unistd.h>      
#include <sys/wait.h>    
#include <vector>
#include <string>
#include <sstream>       
#include <iostream>
#include <algorithm>    
#include <thread>        
using namespace std;
#include <sstream>
#include <cstring>
#include<pthread.h>
#include<omp.h>
#include<atomic>

void serialize_segment(const segment& seg, std::vector<int>& int_buffer, std::vector<char>& char_buffer) {
    // 1. 序列化 segment 的基本 int 成员
    int_buffer.push_back(seg.type);
    int_buffer.push_back(seg.length);
    int_buffer.push_back(seg.total_freq);

    // 2. 序列化 ordered_values (vector<string>)
    int_buffer.push_back(seg.ordered_values.size());
    for (const auto& val : seg.ordered_values) {
        int_buffer.push_back(val.length());
        char_buffer.insert(char_buffer.end(), val.begin(), val.end());
    }

    // 3. 序列化 ordered_freqs (vector<int>)
    int_buffer.push_back(seg.ordered_freqs.size());
    int_buffer.insert(int_buffer.end(), seg.ordered_freqs.begin(), seg.ordered_freqs.end());

    // 4. 序列化 values (unordered_map<string, int>)
    int_buffer.push_back(seg.values.size());
    for (const auto& pair : seg.values) {
        // 序列化 string key
        int_buffer.push_back(pair.first.length());
        char_buffer.insert(char_buffer.end(), pair.first.begin(), pair.first.end());
        // 序列化 int value
        int_buffer.push_back(pair.second);
    }

    // 5. 序列化 freqs (unordered_map<int, int>)
    int_buffer.push_back(seg.freqs.size());
    for (const auto& pair : seg.freqs) {
        int_buffer.push_back(pair.first);  // key
        int_buffer.push_back(pair.second); // value
    }
}

segment deserialize_segment(const std::vector<int>& int_buffer, size_t& int_offset, const std::vector<char>& char_buffer, size_t& char_offset) {
    // 1. 反序列化基本 int 成员
    int type = int_buffer[int_offset++];
    int length = int_buffer[int_offset++];
    segment seg(type, length); // 使用构造函数
    seg.total_freq = int_buffer[int_offset++];

    // 2. 反序列化 ordered_values
    size_t ov_size = int_buffer[int_offset++];
    seg.ordered_values.reserve(ov_size);
    for (size_t i = 0; i < ov_size; ++i) {
        size_t str_len = int_buffer[int_offset++];
        seg.ordered_values.emplace_back(char_buffer.begin() + char_offset, char_buffer.begin() + char_offset + str_len);
        char_offset += str_len;
    }

    // 3. 反序列化 ordered_freqs
    size_t of_size = int_buffer[int_offset++];
    seg.ordered_freqs.assign(int_buffer.begin() + int_offset, int_buffer.begin() + int_offset + of_size);
    int_offset += of_size;

    // 4. 反序列化 values
    size_t v_size = int_buffer[int_offset++];
    seg.values.reserve(v_size);
    for (size_t i = 0; i < v_size; ++i) {
        // 反序列化 string key
        size_t key_len = int_buffer[int_offset++];
        std::string key(char_buffer.begin() + char_offset, char_buffer.begin() + char_offset + key_len);
        char_offset += key_len;
        // 反序列化 int value
        int value = int_buffer[int_offset++];
        seg.values[key] = value;
    }

    // 5. 反序列化 freqs
    size_t f_size = int_buffer[int_offset++];
    seg.freqs.reserve(f_size);
    for (size_t i = 0; i < f_size; ++i) {
        int key = int_buffer[int_offset++];
        int value = int_buffer[int_offset++];
        seg.freqs[key] = value;
    }
    
    return seg;
}


// --- 核心函数：序列化/反序列化单个 PT ---
void serialize_pt(const PT& pt, std::vector<int>& int_buffer, std::vector<char>& char_buffer) {
    // 1. 序列化 content (segment 向量)
    int_buffer.push_back(pt.content.size());
    for (const auto& seg : pt.content) {
        serialize_segment(seg, int_buffer, char_buffer);
    }

    // 2. 序列化 pivot
    int_buffer.push_back(pt.pivot);

    // 3. 序列化 curr_indices
    int_buffer.push_back(pt.curr_indices.size());
    int_buffer.insert(int_buffer.end(), pt.curr_indices.begin(), pt.curr_indices.end());

    // 4. 序列化 max_indices
    int_buffer.push_back(pt.max_indices.size());
    int_buffer.insert(int_buffer.end(), pt.max_indices.begin(), pt.max_indices.end());

    // 5. 序列化 float 类型的概率值 (通过 memcpy 转换成 int)
    int preterm_prob_as_int, prob_as_int;
    static_assert(sizeof(float) == sizeof(int), "Float and int sizes must match for serialization.");
    memcpy(&preterm_prob_as_int, &pt.preterm_prob, sizeof(float));
    memcpy(&prob_as_int, &pt.prob, sizeof(float));
    int_buffer.push_back(preterm_prob_as_int);
    int_buffer.push_back(prob_as_int);
}

PT deserialize_pt(const std::vector<int>& int_buffer, size_t& int_offset, const std::vector<char>& char_buffer, size_t& char_offset) {
    PT pt;

    // 1. 反序列化 content
    if (int_offset >= int_buffer.size()) throw std::runtime_error("Deserialization error: buffer underflow for content size.");
    size_t content_size = int_buffer[int_offset++];
    pt.content.reserve(content_size);
    for (size_t i = 0; i < content_size; ++i) {
        pt.content.push_back(deserialize_segment(int_buffer, int_offset, char_buffer, char_offset));
    }

    // 2. 反序列化 pivot
    if (int_offset >= int_buffer.size()) throw std::runtime_error("Deserialization error: buffer underflow for pivot.");
    pt.pivot = int_buffer[int_offset++];

    // 3. 反序列化 curr_indices
    if (int_offset >= int_buffer.size()) throw std::runtime_error("Deserialization error: buffer underflow for curr_indices size.");
    size_t curr_size = int_buffer[int_offset++];
    if (int_offset + curr_size > int_buffer.size()) throw std::runtime_error("Deserialization error: buffer underflow for curr_indices data.");
    pt.curr_indices.assign(int_buffer.begin() + int_offset, int_buffer.begin() + int_offset + curr_size);
    int_offset += curr_size;

    // 4. 反序列化 max_indices
    if (int_offset >= int_buffer.size()) throw std::runtime_error("Deserialization error: buffer underflow for max_indices size.");
    size_t max_size = int_buffer[int_offset++];
    if (int_offset + max_size > int_buffer.size()) throw std::runtime_error("Deserialization error: buffer underflow for max_indices data.");
    pt.max_indices.assign(int_buffer.begin() + int_offset, int_buffer.begin() + int_offset + max_size);
    int_offset += max_size;

    // 5. 反序列化 float 类型的概率值
    if (int_offset + 1 >= int_buffer.size()) throw std::runtime_error("Deserialization error: buffer underflow for probability data.");
    int preterm_prob_as_int = int_buffer[int_offset++];
    int prob_as_int = int_buffer[int_offset++];
    memcpy(&pt.preterm_prob, &preterm_prob_as_int, sizeof(float));
    memcpy(&pt.prob, &prob_as_int, sizeof(float));

    return pt;
}


// --- 公开接口函数：序列化/反序列化 PT 向量 ---
void serialize_pt_vector(const std::vector<PT>& pts, std::vector<int>& int_buffer, std::vector<char>& char_buffer) {
    // 清空缓冲区，确保从头开始
    int_buffer.clear();
    char_buffer.clear();
    
    // 首先存入向量中 PT 的数量
    int_buffer.push_back(pts.size());
    for (const auto& pt : pts) {
        serialize_pt(pt, int_buffer, char_buffer);
    }
}

std::vector<PT> deserialize_pt_vector(const std::vector<int>& int_buffer, const std::vector<char>& char_buffer) {
    std::vector<PT> pts;
    if (int_buffer.empty()) {
        return pts;
    }

    size_t int_offset = 0;
    size_t char_offset = 0;

    size_t num_pts = int_buffer[int_offset++];
    pts.reserve(num_pts);

    for (size_t i = 0; i < num_pts; ++i) {
        pts.push_back(deserialize_pt(int_buffer, int_offset, char_buffer, char_offset));
    }
    
    // 完整性检查
    if (int_offset != int_buffer.size() || char_offset != char_buffer.size()) {
        // 这通常表示序列化和反序列化逻辑之间存在不匹配
        // 在调试时，可以抛出异常或打印警告
        // throw std::runtime_error("Deserialization warning: buffers not fully consumed.");
    }

    return pts;
}

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext(MPI_Comm comm)
{
    int rank,size;
    int batch_size=4;//每个进程处理4个PT
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    vector<PT> local_batch; 
    vector<PT> local_new_pts;
    int total_to_send;
    std::vector<PT> send_batch;
    if (rank == 0)
    {
        // 1. 准备要分发的 PT 总批次 
        int sum_n = size * batch_size;
        total_to_send = min(sum_n, (int)priority.size());
        
        if (total_to_send > 0) {
            send_batch.insert(send_batch.end(), priority.begin(), priority.begin() + total_to_send);
            priority.erase(priority.begin(), priority.begin() + total_to_send);
        }

        // 2. 计算每个进程应该接收多少个 PT 
        int base = total_to_send / size;
        int remainder = total_to_send % size;
        vector<int> counts(size);
        vector<int> offsets(size, 0); // 初始化为0
        for (int i = 0; i < size; ++i) {
            counts[i] = base + (i < remainder ? 1 : 0);
            if (i > 0) {
                offsets[i] = offsets[i-1] + counts[i-1];
            }
        }

        // 3. 主进程(rank 0)直接获取自己的本地任务批次
        if (counts[0] > 0) {
            local_batch.insert(local_batch.end(), send_batch.begin() + offsets[0], send_batch.begin() + offsets[0] + counts[0]);
        }
        
        // 4. 为其他每个进程准备数据并发送
        for (int proc = 1; proc < size; ++proc) {
            // 如果该进程没有被分配到任务，则跳过
            if (counts[proc] == 0) {
                // 发送一个空任务信号 (两个缓冲区大小都为0)
                int empty_sizes[2] = {0, 0};
                MPI_Send(empty_sizes, 2, MPI_INT, proc, 100, comm);
                continue;
            }

            // a. 为 proc 进程创建一个专属的 PT 批次
            std::vector<PT> batch_for_proc(send_batch.begin() + offsets[proc], send_batch.begin() + offsets[proc] + counts[proc]);

            // b. 使用新的序列化函数，将该批次序列化到两个缓冲区
            std::vector<int> int_buffer;
            std::vector<char> char_buffer;
            serialize_pt_vector(batch_for_proc, int_buffer, char_buffer);

            // c. 发送两个缓冲区的大小
            int sizes[2] = { (int)int_buffer.size(), (int)char_buffer.size() };
            MPI_Send(sizes, 2, MPI_INT, proc, 100, comm);

            // d. 发送缓冲区本身 (仅在非空时发送)
            if (sizes[0] > 0) {
                MPI_Send(int_buffer.data(), sizes[0], MPI_INT, proc, 101, comm);
            }
            if (sizes[1] > 0) {
                MPI_Send(char_buffer.data(), sizes[1], MPI_CHAR, proc, 102, comm);
            }
        }
    } 
    else // 工作进程 (rank != 0) 的接收逻辑
    {
        // a. 接收两个缓冲区的大小
        int sizes[2];
        MPI_Recv(sizes, 2, MPI_INT, 0, 100, comm, MPI_STATUS_IGNORE);

        // 如果收到的是空任务，则 local_batch 保持为空
        if (sizes[0] == 0 && sizes[1] == 0) {
            // 不需要做任何事，local_batch 已经是空的
        } else {
            // b. 根据收到的尺寸，准备接收缓冲区
            std::vector<int> received_int_buffer(sizes[0]);
            std::vector<char> received_char_buffer(sizes[1]);

            // c. 接收两个缓冲区的数据
            if (sizes[0] > 0) {
                MPI_Recv(received_int_buffer.data(), sizes[0], MPI_INT, 0, 101, comm, MPI_STATUS_IGNORE);
            }
            if (sizes[1] > 0) {
                // 注意：MPI_Recv 需要一个可写指针
                MPI_Recv(received_char_buffer.data(), sizes[1], MPI_CHAR, 0, 102, comm, MPI_STATUS_IGNORE);
            }

            // d. 使用新的反序列化函数，直接将结果存入 local_batch
            // (用户要求存入 local_new_pts，但根据上下文，local_batch 是接收初始任务的正确变量)
            local_batch = deserialize_pt_vector(received_int_buffer, received_char_buffer);
        }
                

    }
   // 所有进程都执行这个 for 循环
for(PT& pt : local_batch)
{
    
    Generate(pt);
    vector<PT> temp;
    temp=pt.NewPTs();
    local_new_pts.insert(local_new_pts.end(), temp.begin(), temp.end());
}
    vector<PT> new_pts;
    
    if (rank != 0) 
    {
        // --- 工作进程 (rank > 0) 的发送逻辑 ---
       // 使用序列化函数打包自己的 local_new_pts
        std::vector<int> int_buffer;
        std::vector<char> char_buffer;
        //serialize_pt_vector(local_new_pts, int_buffer, char_buffer);
        try {
        serialize_pt_vector(local_new_pts, int_buffer, char_buffer);
    } catch (const std::exception& e) {
        // 如果序列化函数（特别是带有边界检查的版本）抛出异常，这里会捕获到
        cerr << "Rank " << rank << ": EXCEPTION during serialization: " << e.what() << endl;
        MPI_Abort(comm, 1); // 主动终止所有进程，并给出错误码
    }
        // a. 发送两个缓冲区的大小，让主进程知道要接收多少数据
        int sizes[2] = { (int)int_buffer.size(), (int)char_buffer.size() };
        MPI_Send(sizes, 2, MPI_INT, 0, 300, comm);
        // b. 发送缓冲区本身 (仅在非空时发送)
        if (sizes[0] > 0) {
            MPI_Send(int_buffer.data(), sizes[0], MPI_INT, 0, 301, comm);
        }
        if (sizes[1] > 0) {
            MPI_Send(char_buffer.data(), sizes[1], MPI_CHAR, 0, 302, comm);
        }
    } 
    else // rank == 0
    {
        // --- 主进程 (rank == 0) 的接收逻辑 ---

        // 1. 首先，将自己本地生成的 new_pts 添加到最终结果中
        new_pts = local_new_pts;

        // 2. 循环接收来自所有其他工作进程的数据
        for (int source_rank = 1; source_rank < size; ++source_rank) 
        {
            // a. 接收缓冲区大小
            int sizes[2];
            MPI_Recv(sizes, 2, MPI_INT, source_rank, 300, comm, MPI_STATUS_IGNORE);

            // 如果收到的是空任务，则直接跳过此进程
            if (sizes[0] == 0 && sizes[1] == 0) {
                continue;
            }

            // b. 准备接收缓冲区
            std::vector<int> received_int_buffer(sizes[0]);
            std::vector<char> received_char_buffer(sizes[1]);

            // c. 接收两个缓冲区的数据
            if (sizes[0] > 0) {
                MPI_Recv(received_int_buffer.data(), sizes[0], MPI_INT, source_rank, 301, comm, MPI_STATUS_IGNORE);
            }
            if (sizes[1] > 0) {
                MPI_Recv(received_char_buffer.data(), sizes[1], MPI_CHAR, source_rank, 302, comm, MPI_STATUS_IGNORE);
            }

            // d. 使用反序列化函数，将收到的数据转换成 PT 向量
            std::vector<PT> received_pts = deserialize_pt_vector(received_int_buffer, received_char_buffer);

            // e. 将收到的 PT 集合合并到主进程的 new_pts 列表中
            new_pts.insert(new_pts.end(), received_pts.begin(), received_pts.end());
        }
    
        for (PT pt : new_pts)
        {
            CalProb(pt);
            // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
            for (auto iter = priority.begin(); iter != priority.end(); iter++)
            {
                // 对于非队首和队尾的特殊情况
                if (iter != priority.end() - 1 && iter != priority.begin())
                {
                    // 判定概率
                    if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                    {
                        priority.emplace(iter + 1, pt);
                        break;
                    }
                }
                if (iter == priority.end() - 1)
                {
                    priority.emplace_back(pt);
                    break;
                }
                if (iter == priority.begin() && iter->prob < pt.prob)
                {
                    priority.emplace(iter, pt);
                    break;
                }
            }
        }
    }
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        
        #pragma omp parallel for num_threads(7)\
       schedule(static)\
       reduction(+:total_guesses) reduction(merge:guesses)
       for (int i = 0; i < pt.max_indices[0]; i += 1)
        {   
           
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        
       #pragma omp parallel for num_threads(7)\
       schedule(static)\
       reduction(+:total_guesses) reduction(merge:guesses)
       for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            
            guesses.emplace_back(move(temp));
            total_guesses += 1;
        }
    }
}