
#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
 // 定义了一系列MD5中的具体函数
 // 这四个计算函数是需要你进行SIMD并行化的
 // 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

 /**
  * @brief 使用 Neon intrinsics 实现 F(x, y, z) = ((x & y) | (~x & z))
  * @param vx 包含 4 个 x 操作数的向量 (uint32x4_t)
  * @param vy 包含 4 个 y 操作数的向量 (uint32x4_t)
  * @param vz 包含 4 个 z 操作数的向量 (uint32x4_t)
  * @return 包含 4 个 F(x, y, z) 结果的向量 (uint32x4_t)
  */
static inline uint32x4_t F(uint32x4_t vx, uint32x4_t vy, uint32x4_t vz)
{
	// 计算 (x & y)
	uint32x4_t x_and_y = vandq_u32(vx, vy);
	// 计算 ~x
	uint32x4_t not_x = vmvnq_u32(vx);
	// 计算 (~x & z)
	uint32x4_t not_x_and_z = vandq_u32(not_x, vz);
	// 计算 (x & y) | (~x & z)
	uint32x4_t result = vorrq_u32(x_and_y, not_x_and_z);
	return result;
}
/**
 * @brief 使用 Neon intrinsics 实现 G(x, y, z) = ((x & z) | (y & ~z))
 * @param vx 包含 4 个 x 操作数的向量 (uint32x4_t)
 * @param vy 包含 4 个 y 操作数的向量 (uint32x4_t)
 * @param vz 包含 4 个 z 操作数的向量 (uint32x4_t)
 * @return 包含 4 个 G(x, y, z) 结果的向量 (uint32x4_t)
 */
static inline uint32x4_t G(uint32x4_t vx, uint32x4_t vy, uint32x4_t vz)
{
	uint32x4_t x_and_z = vandq_u32(vx, vz);
	uint32x4_t not_z = vmvnq_u32(vz);
	uint32x4_t y_and_not_z = vandq_u32(vy, not_z);
	uint32x4_t result = vorrq_u32(x_and_z, y_and_not_z);
	return result;
}
/**
 * @brief 使用 Neon intrinsics 实现 H(x, y, z) = (x ^ y ^ z)
 * @param vx 包含 4 个 x 操作数的向量 (uint32x4_t)
 * @param vy 包含 4 个 y 操作数的向量 (uint32x4_t)
 * @param vz 包含 4 个 z 操作数的向量 (uint32x4_t)
 * @return 包含 4 个 H(x, y, z) 结果的向量 (uint32x4_t)
 */
static inline uint32x4_t H(uint32x4_t vx, uint32x4_t vy, uint32x4_t vz) {
	// 计算 x ^ y
	uint32x4_t x_xor_y = veorq_u32(vx, vy);
	// 计算 (x ^ y) ^ z
	uint32x4_t result = veorq_u32(x_xor_y, vz);
	return result;
}
/**
 * @brief 使用 Neon intrinsics 实现 I(x, y, z) = (y ^ (x | ~z))
 * @param vx 包含 4 个 x 操作数的向量 (uint32x4_t)
 * @param vy 包含 4 个 y 操作数的向量 (uint32x4_t)
 * @param vz 包含 4 个 z 操作数的向量 (uint32x4_t)
 * @return 包含 4 个 I(x, y, z) 结果的向量 (uint32x4_t)
 */
static inline uint32x4_t I(uint32x4_t vx, uint32x4_t vy, uint32x4_t vz) {
	// 计算 ~z
	uint32x4_t not_z = vmvnq_u32(vz);
	// 计算 (x | ~z)
	uint32x4_t x_or_not_z = vorrq_u32(vx, not_z);
	// 计算 y ^ (x | ~z)
	uint32x4_t result = veorq_u32(vy, x_or_not_z);
	return result;
}

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
 // 定义了一系列MD5中的具体函数
 // 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
 // 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
#include <arm_neon.h>
static inline uint32x4_t ROTATELEFT(uint32x4_t vnum, const int n) {
	// 原始(((num) << (n)) | ((num) >> (32-(n))))
	if (n == 0) {
		return vnum;
	}
	uint32x4_t left_shifted = vshlq_n_u32(vnum, n); // Neon移位通常接受int
	// 计算右移部分: vnum >> (32 - n)
	uint32x4_t right_shifted = vshrq_n_u32(vnum, 32 - n);
	// 合并结果: (vnum << n) | (vnum >> (32 - n))
	// 使用 vorrq_u32 将左移和右移的结果按位或
	uint32x4_t result = vorrq_u32(left_shifted, right_shifted);
	return result;
}
/**
 * @brief 使用 Neon intrinsics 实现 FF 宏定义的操作，一次处理 4 组 32 位数据。
 *        对应宏: #define FF(a, b, c, d, x, s, ac) { \
 *                   (a) += F ((b), (c), (d)) + (x) + ac; \
 *                   (a) = ROTATELEFT ((a), (s)); \
 *                   (a) += (b); \
 *                 }
 * @param va 对应宏中的 'a' (输入向量，函数内部会对其进行计算并返回新值)
 * @param vb 对应宏中的 'b' (输入向量)
 * @param vc 对应宏中的 'c' (输入向量)
 * @param vd 对应宏中的 'd' (输入向量)
 * @param vx 对应宏中的 'x' (输入向量，通常是消息块)
 * @param s  对应宏中的 's' (循环左移的位数，标量)
 * @param ac 对应宏中的 'ac' (加法常量，32位标量)
 * @return 计算后的 'a' 向量 (uint32x4_t)
 */
static inline uint32x4_t FF(uint32x4_t va, uint32x4_t vb, uint32x4_t vc, uint32x4_t vd, uint32x4_t vx, const int s, uint32_t ac) {

	// 1. 计算 F(b, c, d)
	uint32x4_t f_result = F(vb, vc, vd);
	// 2. 将加法常量 ac 复制到向量的所有通道
	uint32x4_t vac = vdupq_n_u32(ac);

	// 3. 计算 a += F(b, c, d) + x + ac
	// 分步进行加法：
	// temp = F(b, c, d) + x
	uint32x4_t temp_sum1 = vaddq_u32(f_result, vx);
	// temp = temp + ac
	uint32x4_t temp_sum2 = vaddq_u32(temp_sum1, vac);
	// a = a + temp
	va = vaddq_u32(va, temp_sum2);

	// 4. 计算 a = ROTATELEFT(a, s)

	va = ROTATELEFT(va, s);

	// 5. 计算 a += b
	va = vaddq_u32(va, vb);

	// 6. 返回最终更新的 a 向量
	return va;
}
static inline uint32x4_t GG(uint32x4_t va, uint32x4_t vb, uint32x4_t vc, uint32x4_t vd, uint32x4_t vx, const int s, uint32_t ac) {

	// 1. 计算 G(b, c, d)
	uint32x4_t g_result = G(vb, vc, vd);
	// 2. 将加法常量 ac 复制到向量的所有通道
	uint32x4_t vac = vdupq_n_u32(ac);
	// 3. 计算 a += G(b, c, d) + x + ac
	// 分步进行加法：
	// temp = G(b, c, d) + x
	uint32x4_t temp_sum1 = vaddq_u32(g_result, vx);
	// temp = temp + ac
	uint32x4_t temp_sum2 = vaddq_u32(temp_sum1, vac);
	// a = a + temp
	va = vaddq_u32(va, temp_sum2);

	// 4. 计算 a = ROTATELEFT(a, s)
	va = ROTATELEFT(va, s); // 调用你上面定义的ROTATELEFT函数

	// 5. 计算 a += b
	va = vaddq_u32(va, vb);

	// 6. 返回最终更新的 a 向量
	return va;
}
static inline uint32x4_t HH(uint32x4_t va, uint32x4_t vb, uint32x4_t vc, uint32x4_t vd, uint32x4_t vx, const int s, uint32_t ac) {

	// 1. 计算 H(b, c, d)
	uint32x4_t h_result = H(vb, vc, vd);

	// 2. 将加法常量 ac 复制到向量的所有通道
	uint32x4_t vac = vdupq_n_u32(ac);

	// 3. 计算 a += H(b, c, d) + x + ac
	uint32x4_t temp_sum1 = vaddq_u32(h_result, vx);
	uint32x4_t temp_sum2 = vaddq_u32(temp_sum1, vac);
	va = vaddq_u32(va, temp_sum2);

	// 4. 计算 a = ROTATELEFT(a, s)
	va = ROTATELEFT(va, s);

	// 5. 计算 a += b
	va = vaddq_u32(va, vb);

	// 6. 返回最终更新的 a 向量
	return va;
}
static inline uint32x4_t II(uint32x4_t va, uint32x4_t vb, uint32x4_t vc, uint32x4_t vd, uint32x4_t vx, int s, uint32_t ac) {

	// 1. 计算 I(b, c, d)
	uint32x4_t i_result = I(vb, vc, vd);

	// 2. 将加法常量 ac 复制到向量的所有通道
	uint32x4_t vac = vdupq_n_u32(ac);

	// 3. 计算 a += I(b, c, d) + x + ac
	uint32x4_t temp_sum1 = vaddq_u32(i_result, vx);
	uint32x4_t temp_sum2 = vaddq_u32(temp_sum1, vac);
	va = vaddq_u32(va, temp_sum2);

	// 4. 计算 a = ROTATELEF
	va = ROTATELEFT(va, s);
	// 5. 计算 a += b
	va = vaddq_u32(va, vb);

	// 6. 返回最终更新的 a 向量
	return va;
}
void MD5Hash(const string input[4], bit32 state[4][4]);