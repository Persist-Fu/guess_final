#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <algorithm>
using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte* StringProcess(string input, int* n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte* blocks = (Byte*)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits = 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte* paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}


/**
 * MD5Hash: 将4个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(const string input[4], bit32 state[4][4])
{

	Byte* paddedMessage1, * paddedMessage2, * paddedMessage3, * paddedMessage4;
	int* messageLength1 = new int[1];
	for (int i = 0; i < 1; i += 1)
	{
		paddedMessage1 = StringProcess(input[0], &messageLength1[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength1[i] == messageLength1[0]);
	}
	int n_blocks1 = messageLength1[0] / 64;

	int* messageLength2 = new int[1];
	for (int i = 0; i < 1; i += 1)
	{
		paddedMessage2 = StringProcess(input[1], &messageLength2[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength2[i] == messageLength2[0]);
	}
	int n_blocks2 = messageLength2[0] / 64;

	int* messageLength3 = new int[1];
	for (int i = 0; i < 1; i += 1)
	{
		paddedMessage3 = StringProcess(input[2], &messageLength3[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength3[i] == messageLength3[0]);
	}
	int n_blocks3 = messageLength3[0] / 64;

	int* messageLength4 = new int[1];
	for (int i = 0; i < 1; i += 1)
	{
		paddedMessage4 = StringProcess(input[3], &messageLength4[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength4[i] == messageLength4[0]);
	}
	int n_blocks4 = messageLength4[0] / 64;
	// bit32* state= new bit32[4];
	for (int kk = 0; kk < 4; kk++)
	{
		state[kk][0] = 0x67452301;
		state[kk][1] = 0xefcdab89;
		state[kk][2] = 0x98badcfe;
		state[kk][3] = 0x10325476;
	}

	//找出最小块数
	int aaa[4] = { n_blocks1,n_blocks2,n_blocks3,n_blocks4 };
	int min_n_blocks = aaa[0];
	for (int pi = 1; pi <= 3; pi++)
	{
		if (aaa[pi] < min_n_blocks)
		{
			min_n_blocks = aaa[pi];
		}
	}
	bit32 x[4][16];
	// 逐block地更新state
	for (int i = 0; i < min_n_blocks; i += 1)
	{
		// 下面的处理，在理解上较为复杂
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[0][i1] = (paddedMessage1[4 * i1 + i * 64]) |
				(paddedMessage1[4 * i1 + 1 + i * 64] << 8) |
				(paddedMessage1[4 * i1 + 2 + i * 64] << 16) |
				(paddedMessage1[4 * i1 + 3 + i * 64] << 24);
		}
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[1][i1] = (paddedMessage2[4 * i1 + i * 64]) |
				(paddedMessage2[4 * i1 + 1 + i * 64] << 8) |
				(paddedMessage2[4 * i1 + 2 + i * 64] << 16) |
				(paddedMessage2[4 * i1 + 3 + i * 64] << 24);
		}
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[2][i1] = (paddedMessage3[4 * i1 + i * 64]) |
				(paddedMessage3[4 * i1 + 1 + i * 64] << 8) |
				(paddedMessage3[4 * i1 + 2 + i * 64] << 16) |
				(paddedMessage3[4 * i1 + 3 + i * 64] << 24);
		}
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[3][i1] = (paddedMessage4[4 * i1 + i * 64]) |
				(paddedMessage4[4 * i1 + 1 + i * 64] << 8) |
				(paddedMessage4[4 * i1 + 2 + i * 64] << 16) |
				(paddedMessage4[4 * i1 + 3 + i * 64] << 24);
		}


		uint32_t a_values[4] = { state[0][0],state[1][0],state[2][0], state[3][0] };
		uint32_t b_values[4] = { state[0][1],state[1][1],state[2][1],state[3][1] };
		uint32_t c_values[4] = { state[0][2],state[1][2],state[2][2], state[3][2] };
		uint32_t d_values[4] = { state[0][3],state[1][3],state[2][3], state[3][3] };

		uint32x4_t va = vld1q_u32(a_values);
		uint32x4_t vb = vld1q_u32(b_values);
		uint32x4_t vc = vld1q_u32(c_values);
		uint32x4_t vd = vld1q_u32(d_values);

		uint32_t x0_values[4] = { x[0][0],x[1][0],x[2][0], x[3][0] };
		uint32_t x1_values[4] = { x[0][1],x[1][1],x[2][1], x[3][1] };
		uint32_t x2_values[4] = { x[0][2],x[1][2],x[2][2], x[3][2] };
		uint32_t x3_values[4] = { x[0][3],x[1][3],x[2][3], x[3][3] };
		uint32_t x4_values[4] = { x[0][4],x[1][4],x[2][4], x[3][4] };
		uint32_t x5_values[4] = { x[0][5],x[1][5],x[2][5], x[3][5] };
		uint32_t x6_values[4] = { x[0][6],x[1][6],x[2][6], x[3][6] };
		uint32_t x7_values[4] = { x[0][7],x[1][7],x[2][7], x[3][7] };
		uint32_t x8_values[4] = { x[0][8],x[1][8],x[2][8], x[3][8] };
		uint32_t x9_values[4] = { x[0][9],x[1][9],x[2][9], x[3][9] };
		uint32_t x10_values[4] = { x[0][10],x[1][10],x[2][10], x[3][10] };
		uint32_t x11_values[4] = { x[0][11],x[1][11],x[2][11], x[3][11] };
		uint32_t x12_values[4] = { x[0][12],x[1][12],x[2][12], x[3][12] };
		uint32_t x13_values[4] = { x[0][13],x[1][13],x[2][13], x[3][13] };
		uint32_t x14_values[4] = { x[0][14],x[1][14],x[2][14], x[3][14] };
		uint32_t x15_values[4] = { x[0][15],x[1][15],x[2][15], x[3][15] };

		uint32x4_t x0 = vld1q_u32(x0_values);
		uint32x4_t x1 = vld1q_u32(x1_values);
		uint32x4_t x2 = vld1q_u32(x2_values);
		uint32x4_t x3 = vld1q_u32(x3_values);
		uint32x4_t x4 = vld1q_u32(x4_values);
		uint32x4_t x5 = vld1q_u32(x5_values);
		uint32x4_t x6 = vld1q_u32(x6_values);
		uint32x4_t x7 = vld1q_u32(x7_values);
		uint32x4_t x8 = vld1q_u32(x8_values);
		uint32x4_t x9 = vld1q_u32(x9_values);
		uint32x4_t x10 = vld1q_u32(x10_values);
		uint32x4_t x11 = vld1q_u32(x11_values);
		uint32x4_t x12 = vld1q_u32(x12_values);
		uint32x4_t x13 = vld1q_u32(x13_values);
		uint32x4_t x14 = vld1q_u32(x14_values);
		uint32x4_t x15 = vld1q_u32(x15_values);


		auto start = system_clock::now();
		/* Round 1 */
		va = FF(va, vb, vc, vd, x0, s11, 0xd76aa478);
		vd = FF(vd, va, vb, vc, x1, s12, 0xe8c7b756);
		vc = FF(vc, vd, va, vb, x2, s13, 0x242070db);
		vb = FF(vb, vc, vd, va, x3, s14, 0xc1bdceee);
		va = FF(va, vb, vc, vd, x4, s11, 0xf57c0faf);
		vd = FF(vd, va, vb, vc, x5, s12, 0x4787c62a);
		vc = FF(vc, vd, va, vb, x6, s13, 0xa8304613);
		vb = FF(vb, vc, vd, va, x7, s14, 0xfd469501);
		va = FF(va, vb, vc, vd, x8, s11, 0x698098d8);
		vd = FF(vd, va, vb, vc, x9, s12, 0x8b44f7af);
		vc = FF(vc, vd, va, vb, x10, s13, 0xffff5bb1);
		vb = FF(vb, vc, vd, va, x11, s14, 0x895cd7be);
		va = FF(va, vb, vc, vd, x12, s11, 0x6b901122);
		vd = FF(vd, va, vb, vc, x13, s12, 0xfd987193);
		vc = FF(vc, vd, va, vb, x14, s13, 0xa679438e);
		vb = FF(vb, vc, vd, va, x15, s14, 0x49b40821);

		/* Round 2 */
		va = GG(va, vb, vc, vd, x1, s21, 0xf61e2562);
		vd = GG(vd, va, vb, vc, x6, s22, 0xc040b340);
		vc = GG(vc, vd, va, vb, x11, s23, 0x265e5a51);
		vb = GG(vb, vc, vd, va, x0, s24, 0xe9b6c7aa);
		va = GG(va, vb, vc, vd, x5, s21, 0xd62f105d);
		vd = GG(vd, va, vb, vc, x10, s22, 0x2441453);
		vc = GG(vc, vd, va, vb, x15, s23, 0xd8a1e681);
		vb = GG(vb, vc, vd, va, x4, s24, 0xe7d3fbc8);
		va = GG(va, vb, vc, vd, x9, s21, 0x21e1cde6);
		vd = GG(vd, va, vb, vc, x14, s22, 0xc33707d6);
		vc = GG(vc, vd, va, vb, x3, s23, 0xf4d50d87);
		vb = GG(vb, vc, vd, va, x8, s24, 0x455a14ed);
		va = GG(va, vb, vc, vd, x13, s21, 0xa9e3e905);
		vd = GG(vd, va, vb, vc, x2, s22, 0xfcefa3f8);
		vc = GG(vc, vd, va, vb, x7, s23, 0x676f02d9);
		vb = GG(vb, vc, vd, va, x12, s24, 0x8d2a4c8a);

		/* Round 3 */
		va = HH(va, vb, vc, vd, x5, s31, 0xfffa3942);
		vd = HH(vd, va, vb, vc, x8, s32, 0x8771f681);
		vc = HH(vc, vd, va, vb, x11, s33, 0x6d9d6122);
		vb = HH(vb, vc, vd, va, x14, s34, 0xfde5380c);
		va = HH(va, vb, vc, vd, x1, s31, 0xa4beea44);
		vd = HH(vd, va, vb, vc, x4, s32, 0x4bdecfa9);
		vc = HH(vc, vd, va, vb, x7, s33, 0xf6bb4b60);
		vb = HH(vb, vc, vd, va, x10, s34, 0xbebfbc70);
		va = HH(va, vb, vc, vd, x13, s31, 0x289b7ec6);
		vd = HH(vd, va, vb, vc, x0, s32, 0xeaa127fa);
		vc = HH(vc, vd, va, vb, x3, s33, 0xd4ef3085);
		vb = HH(vb, vc, vd, va, x6, s34, 0x4881d05);
		va = HH(va, vb, vc, vd, x9, s31, 0xd9d4d039);
		vd = HH(vd, va, vb, vc, x12, s32, 0xe6db99e5);
		vc = HH(vc, vd, va, vb, x15, s33, 0x1fa27cf8);
		vb = HH(vb, vc, vd, va, x2, s34, 0xc4ac5665);

		/* Round 4 */
		va = II(va, vb, vc, vd, x0, s41, 0xf4292244);
		vd = II(vd, va, vb, vc, x7, s42, 0x432aff97);
		vc = II(vc, vd, va, vb, x14, s43, 0xab9423a7);
		vb = II(vb, vc, vd, va, x5, s44, 0xfc93a039);
		va = II(va, vb, vc, vd, x12, s41, 0x655b59c3);
		vd = II(vd, va, vb, vc, x3, s42, 0x8f0ccc92);
		vc = II(vc, vd, va, vb, x10, s43, 0xffeff47d);
		vb = II(vb, vc, vd, va, x1, s44, 0x85845dd1);
		va = II(va, vb, vc, vd, x8, s41, 0x6fa87e4f);
		vd = II(vd, va, vb, vc, x15, s42, 0xfe2ce6e0);
		vc = II(vc, vd, va, vb, x6, s43, 0xa3014314);
		vb = II(vb, vc, vd, va, x13, s44, 0x4e0811a1);
		va = II(va, vb, vc, vd, x4, s41, 0xf7537e82);
		vd = II(vd, va, vb, vc, x11, s42, 0xbd3af235);
		vc = II(vc, vd, va, vb, x2, s43, 0x2ad7d2bb);
		vb = II(vb, vc, vd, va, x9, s44, 0xeb86d391);

		state[0][0] += vgetq_lane_u32(va, 0);
		state[0][1] += vgetq_lane_u32(vb, 0);
		state[0][2] += vgetq_lane_u32(vc, 0);
		state[0][3] += vgetq_lane_u32(vd, 0);
		state[1][0] += vgetq_lane_u32(va, 1);
		state[1][1] += vgetq_lane_u32(vb, 1);
		state[1][2] += vgetq_lane_u32(vc, 1);
		state[1][3] += vgetq_lane_u32(vd, 1);
		state[2][0] += vgetq_lane_u32(va, 2);
		state[2][1] += vgetq_lane_u32(vb, 2);
		state[2][2] += vgetq_lane_u32(vc, 2);
		state[2][3] += vgetq_lane_u32(vd, 2);
		state[3][0] += vgetq_lane_u32(va, 3);
		state[3][1] += vgetq_lane_u32(vb, 3);
		state[3][2] += vgetq_lane_u32(vc, 3);
		state[3][3] += vgetq_lane_u32(vd, 3);
	}
	if (n_blocks1 > min_n_blocks)
	{
		for (int i = min_n_blocks; i < n_blocks1; i += 1)
		{
			for (int i1 = 0; i1 < 16; ++i1)
			{
				x[0][i1] = (paddedMessage1[4 * i1 + i * 64]) |
					(paddedMessage1[4 * i1 + 1 + i * 64] << 8) |
					(paddedMessage1[4 * i1 + 2 + i * 64] << 16) |
					(paddedMessage1[4 * i1 + 3 + i * 64] << 24);
			}
			uint32x4_t va = vdupq_n_u32(state[0][0]);
			uint32x4_t vb = vdupq_n_u32(state[0][1]);
			uint32x4_t vc = vdupq_n_u32(state[0][2]);
			uint32x4_t vd = vdupq_n_u32(state[0][3]);
			uint32x4_t x0 = vdupq_n_u32(x[0][0]);
			uint32x4_t x1 = vdupq_n_u32(x[0][1]);
			uint32x4_t x2 = vdupq_n_u32(x[0][2]);
			uint32x4_t x3 = vdupq_n_u32(x[0][3]);
			uint32x4_t x4 = vdupq_n_u32(x[0][4]);
			uint32x4_t x5 = vdupq_n_u32(x[0][5]);
			uint32x4_t x6 = vdupq_n_u32(x[0][6]);
			uint32x4_t x7 = vdupq_n_u32(x[0][7]);
			uint32x4_t x8 = vdupq_n_u32(x[0][8]);
			uint32x4_t x9 = vdupq_n_u32(x[0][9]);
			uint32x4_t x10 = vdupq_n_u32(x[0][10]);
			uint32x4_t x11 = vdupq_n_u32(x[0][11]);
			uint32x4_t x12 = vdupq_n_u32(x[0][12]);
			uint32x4_t x13 = vdupq_n_u32(x[0][13]);
			uint32x4_t x14 = vdupq_n_u32(x[0][14]);
			uint32x4_t x15 = vdupq_n_u32(x[0][15]);
			va = FF(va, vb, vc, vd, x0, s11, 0xd76aa478);
			vd = FF(vd, va, vb, vc, x1, s12, 0xe8c7b756);
			vc = FF(vc, vd, va, vb, x2, s13, 0x242070db);
			vb = FF(vb, vc, vd, va, x3, s14, 0xc1bdceee);
			va = FF(va, vb, vc, vd, x4, s11, 0xf57c0faf);
			vd = FF(vd, va, vb, vc, x5, s12, 0x4787c62a);
			vc = FF(vc, vd, va, vb, x6, s13, 0xa8304613);
			vb = FF(vb, vc, vd, va, x7, s14, 0xfd469501);
			va = FF(va, vb, vc, vd, x8, s11, 0x698098d8);
			vd = FF(vd, va, vb, vc, x9, s12, 0x8b44f7af);
			vc = FF(vc, vd, va, vb, x10, s13, 0xffff5bb1);
			vb = FF(vb, vc, vd, va, x11, s14, 0x895cd7be);
			va = FF(va, vb, vc, vd, x12, s11, 0x6b901122);
			vd = FF(vd, va, vb, vc, x13, s12, 0xfd987193);
			vc = FF(vc, vd, va, vb, x14, s13, 0xa679438e);
			vb = FF(vb, vc, vd, va, x15, s14, 0x49b40821);

			/* Round 2 */
			va = GG(va, vb, vc, vd, x1, s21, 0xf61e2562);
			vd = GG(vd, va, vb, vc, x6, s22, 0xc040b340);
			vc = GG(vc, vd, va, vb, x11, s23, 0x265e5a51);
			vb = GG(vb, vc, vd, va, x0, s24, 0xe9b6c7aa);
			va = GG(va, vb, vc, vd, x5, s21, 0xd62f105d);
			vd = GG(vd, va, vb, vc, x10, s22, 0x2441453);
			vc = GG(vc, vd, va, vb, x15, s23, 0xd8a1e681);
			vb = GG(vb, vc, vd, va, x4, s24, 0xe7d3fbc8);
			va = GG(va, vb, vc, vd, x9, s21, 0x21e1cde6);
			vd = GG(vd, va, vb, vc, x14, s22, 0xc33707d6);
			vc = GG(vc, vd, va, vb, x3, s23, 0xf4d50d87);
			vb = GG(vb, vc, vd, va, x8, s24, 0x455a14ed);
			va = GG(va, vb, vc, vd, x13, s21, 0xa9e3e905);
			vd = GG(vd, va, vb, vc, x2, s22, 0xfcefa3f8);
			vc = GG(vc, vd, va, vb, x7, s23, 0x676f02d9);
			vb = GG(vb, vc, vd, va, x12, s24, 0x8d2a4c8a);

			/* Round 3 */
			va = HH(va, vb, vc, vd, x5, s31, 0xfffa3942);
			vd = HH(vd, va, vb, vc, x8, s32, 0x8771f681);
			vc = HH(vc, vd, va, vb, x11, s33, 0x6d9d6122);
			vb = HH(vb, vc, vd, va, x14, s34, 0xfde5380c);
			va = HH(va, vb, vc, vd, x1, s31, 0xa4beea44);
			vd = HH(vd, va, vb, vc, x4, s32, 0x4bdecfa9);
			vc = HH(vc, vd, va, vb, x7, s33, 0xf6bb4b60);
			vb = HH(vb, vc, vd, va, x10, s34, 0xbebfbc70);
			va = HH(va, vb, vc, vd, x13, s31, 0x289b7ec6);
			vd = HH(vd, va, vb, vc, x0, s32, 0xeaa127fa);
			vc = HH(vc, vd, va, vb, x3, s33, 0xd4ef3085);
			vb = HH(vb, vc, vd, va, x6, s34, 0x4881d05);
			va = HH(va, vb, vc, vd, x9, s31, 0xd9d4d039);
			vd = HH(vd, va, vb, vc, x12, s32, 0xe6db99e5);
			vc = HH(vc, vd, va, vb, x15, s33, 0x1fa27cf8);
			vb = HH(vb, vc, vd, va, x2, s34, 0xc4ac5665);

			/* Round 4 */
			va = II(va, vb, vc, vd, x0, s41, 0xf4292244);
			vd = II(vd, va, vb, vc, x7, s42, 0x432aff97);
			vc = II(vc, vd, va, vb, x14, s43, 0xab9423a7);
			vb = II(vb, vc, vd, va, x5, s44, 0xfc93a039);
			va = II(va, vb, vc, vd, x12, s41, 0x655b59c3);
			vd = II(vd, va, vb, vc, x3, s42, 0x8f0ccc92);
			vc = II(vc, vd, va, vb, x10, s43, 0xffeff47d);
			vb = II(vb, vc, vd, va, x1, s44, 0x85845dd1);
			va = II(va, vb, vc, vd, x8, s41, 0x6fa87e4f);
			vd = II(vd, va, vb, vc, x15, s42, 0xfe2ce6e0);
			vc = II(vc, vd, va, vb, x6, s43, 0xa3014314);
			vb = II(vb, vc, vd, va, x13, s44, 0x4e0811a1);
			va = II(va, vb, vc, vd, x4, s41, 0xf7537e82);
			vd = II(vd, va, vb, vc, x11, s42, 0xbd3af235);
			vc = II(vc, vd, va, vb, x2, s43, 0x2ad7d2bb);
			vb = II(vb, vc, vd, va, x9, s44, 0xeb86d391);

			state[0][0] += vgetq_lane_u32(va, 0);
			state[0][1] += vgetq_lane_u32(vb, 0);
			state[0][2] += vgetq_lane_u32(vc, 0);
			state[0][3] += vgetq_lane_u32(vd, 0);
		}
	}

	if (n_blocks2 > min_n_blocks)
	{
		for (int i = min_n_blocks; i < n_blocks2; i += 1)
		{
			for (int i1 = 0; i1 < 16; ++i1)
			{
				x[1][i1] = (paddedMessage2[4 * i1 + i * 64]) |
					(paddedMessage2[4 * i1 + 1 + i * 64] << 8) |
					(paddedMessage2[4 * i1 + 2 + i * 64] << 16) |
					(paddedMessage2[4 * i1 + 3 + i * 64] << 24);
			}
			uint32x4_t va = vdupq_n_u32(state[1][0]);
			uint32x4_t vb = vdupq_n_u32(state[1][1]);
			uint32x4_t vc = vdupq_n_u32(state[1][2]);
			uint32x4_t vd = vdupq_n_u32(state[1][3]);
			uint32x4_t x0 = vdupq_n_u32(x[1][0]);
			uint32x4_t x1 = vdupq_n_u32(x[1][1]);
			uint32x4_t x2 = vdupq_n_u32(x[1][2]);
			uint32x4_t x3 = vdupq_n_u32(x[1][3]);
			uint32x4_t x4 = vdupq_n_u32(x[1][4]);
			uint32x4_t x5 = vdupq_n_u32(x[1][5]);
			uint32x4_t x6 = vdupq_n_u32(x[1][6]);
			uint32x4_t x7 = vdupq_n_u32(x[1][7]);
			uint32x4_t x8 = vdupq_n_u32(x[1][8]);
			uint32x4_t x9 = vdupq_n_u32(x[1][9]);
			uint32x4_t x10 = vdupq_n_u32(x[1][10]);
			uint32x4_t x11 = vdupq_n_u32(x[1][11]);
			uint32x4_t x12 = vdupq_n_u32(x[1][12]);
			uint32x4_t x13 = vdupq_n_u32(x[1][13]);
			uint32x4_t x14 = vdupq_n_u32(x[1][14]);
			uint32x4_t x15 = vdupq_n_u32(x[1][15]);

			va = FF(va, vb, vc, vd, x0, s11, 0xd76aa478);
			vd = FF(vd, va, vb, vc, x1, s12, 0xe8c7b756);
			vc = FF(vc, vd, va, vb, x2, s13, 0x242070db);
			vb = FF(vb, vc, vd, va, x3, s14, 0xc1bdceee);
			va = FF(va, vb, vc, vd, x4, s11, 0xf57c0faf);
			vd = FF(vd, va, vb, vc, x5, s12, 0x4787c62a);
			vc = FF(vc, vd, va, vb, x6, s13, 0xa8304613);
			vb = FF(vb, vc, vd, va, x7, s14, 0xfd469501);
			va = FF(va, vb, vc, vd, x8, s11, 0x698098d8);
			vd = FF(vd, va, vb, vc, x9, s12, 0x8b44f7af);
			vc = FF(vc, vd, va, vb, x10, s13, 0xffff5bb1);
			vb = FF(vb, vc, vd, va, x11, s14, 0x895cd7be);
			va = FF(va, vb, vc, vd, x12, s11, 0x6b901122);
			vd = FF(vd, va, vb, vc, x13, s12, 0xfd987193);
			vc = FF(vc, vd, va, vb, x14, s13, 0xa679438e);
			vb = FF(vb, vc, vd, va, x15, s14, 0x49b40821);

			/* Round 2 */
			va = GG(va, vb, vc, vd, x1, s21, 0xf61e2562);
			vd = GG(vd, va, vb, vc, x6, s22, 0xc040b340);
			vc = GG(vc, vd, va, vb, x11, s23, 0x265e5a51);
			vb = GG(vb, vc, vd, va, x0, s24, 0xe9b6c7aa);
			va = GG(va, vb, vc, vd, x5, s21, 0xd62f105d);
			vd = GG(vd, va, vb, vc, x10, s22, 0x2441453);
			vc = GG(vc, vd, va, vb, x15, s23, 0xd8a1e681);
			vb = GG(vb, vc, vd, va, x4, s24, 0xe7d3fbc8);
			va = GG(va, vb, vc, vd, x9, s21, 0x21e1cde6);
			vd = GG(vd, va, vb, vc, x14, s22, 0xc33707d6);
			vc = GG(vc, vd, va, vb, x3, s23, 0xf4d50d87);
			vb = GG(vb, vc, vd, va, x8, s24, 0x455a14ed);
			va = GG(va, vb, vc, vd, x13, s21, 0xa9e3e905);
			vd = GG(vd, va, vb, vc, x2, s22, 0xfcefa3f8);
			vc = GG(vc, vd, va, vb, x7, s23, 0x676f02d9);
			vb = GG(vb, vc, vd, va, x12, s24, 0x8d2a4c8a);

			/* Round 3 */
			va = HH(va, vb, vc, vd, x5, s31, 0xfffa3942);
			vd = HH(vd, va, vb, vc, x8, s32, 0x8771f681);
			vc = HH(vc, vd, va, vb, x11, s33, 0x6d9d6122);
			vb = HH(vb, vc, vd, va, x14, s34, 0xfde5380c);
			va = HH(va, vb, vc, vd, x1, s31, 0xa4beea44);
			vd = HH(vd, va, vb, vc, x4, s32, 0x4bdecfa9);
			vc = HH(vc, vd, va, vb, x7, s33, 0xf6bb4b60);
			vb = HH(vb, vc, vd, va, x10, s34, 0xbebfbc70);
			va = HH(va, vb, vc, vd, x13, s31, 0x289b7ec6);
			vd = HH(vd, va, vb, vc, x0, s32, 0xeaa127fa);
			vc = HH(vc, vd, va, vb, x3, s33, 0xd4ef3085);
			vb = HH(vb, vc, vd, va, x6, s34, 0x4881d05);
			va = HH(va, vb, vc, vd, x9, s31, 0xd9d4d039);
			vd = HH(vd, va, vb, vc, x12, s32, 0xe6db99e5);
			vc = HH(vc, vd, va, vb, x15, s33, 0x1fa27cf8);
			vb = HH(vb, vc, vd, va, x2, s34, 0xc4ac5665);

			/* Round 4 */
			va = II(va, vb, vc, vd, x0, s41, 0xf4292244);
			vd = II(vd, va, vb, vc, x7, s42, 0x432aff97);
			vc = II(vc, vd, va, vb, x14, s43, 0xab9423a7);
			vb = II(vb, vc, vd, va, x5, s44, 0xfc93a039);
			va = II(va, vb, vc, vd, x12, s41, 0x655b59c3);
			vd = II(vd, va, vb, vc, x3, s42, 0x8f0ccc92);
			vc = II(vc, vd, va, vb, x10, s43, 0xffeff47d);
			vb = II(vb, vc, vd, va, x1, s44, 0x85845dd1);
			va = II(va, vb, vc, vd, x8, s41, 0x6fa87e4f);
			vd = II(vd, va, vb, vc, x15, s42, 0xfe2ce6e0);
			vc = II(vc, vd, va, vb, x6, s43, 0xa3014314);
			vb = II(vb, vc, vd, va, x13, s44, 0x4e0811a1);
			va = II(va, vb, vc, vd, x4, s41, 0xf7537e82);
			vd = II(vd, va, vb, vc, x11, s42, 0xbd3af235);
			vc = II(vc, vd, va, vb, x2, s43, 0x2ad7d2bb);
			vb = II(vb, vc, vd, va, x9, s44, 0xeb86d391);

			state[1][0] += vgetq_lane_u32(va, 0);
			state[1][1] += vgetq_lane_u32(vb, 0);
			state[1][2] += vgetq_lane_u32(vc, 0);
			state[1][3] += vgetq_lane_u32(vd, 0);
		}
	}
	if (n_blocks3 > min_n_blocks)
	{
		for (int i = min_n_blocks; i < n_blocks3; i += 1)
		{
			for (int i1 = 0; i1 < 16; ++i1)
			{
				x[2][i1] = (paddedMessage3[4 * i1 + i * 64]) |
					(paddedMessage3[4 * i1 + 1 + i * 64] << 8) |
					(paddedMessage3[4 * i1 + 2 + i * 64] << 16) |
					(paddedMessage3[4 * i1 + 3 + i * 64] << 24);
			}
			uint32x4_t va = vdupq_n_u32(state[2][0]);
			uint32x4_t vb = vdupq_n_u32(state[2][1]);
			uint32x4_t vc = vdupq_n_u32(state[2][2]);
			uint32x4_t vd = vdupq_n_u32(state[2][3]);
			uint32x4_t x0 = vdupq_n_u32(x[2][0]);
			uint32x4_t x1 = vdupq_n_u32(x[2][1]);
			uint32x4_t x2 = vdupq_n_u32(x[2][2]);
			uint32x4_t x3 = vdupq_n_u32(x[2][3]);
			uint32x4_t x4 = vdupq_n_u32(x[2][4]);
			uint32x4_t x5 = vdupq_n_u32(x[2][5]);
			uint32x4_t x6 = vdupq_n_u32(x[2][6]);
			uint32x4_t x7 = vdupq_n_u32(x[2][7]);
			uint32x4_t x8 = vdupq_n_u32(x[2][8]);
			uint32x4_t x9 = vdupq_n_u32(x[2][9]);
			uint32x4_t x10 = vdupq_n_u32(x[2][10]);
			uint32x4_t x11 = vdupq_n_u32(x[2][11]);
			uint32x4_t x12 = vdupq_n_u32(x[2][12]);
			uint32x4_t x13 = vdupq_n_u32(x[2][13]);
			uint32x4_t x14 = vdupq_n_u32(x[2][14]);
			uint32x4_t x15 = vdupq_n_u32(x[2][15]);
			va = FF(va, vb, vc, vd, x0, s11, 0xd76aa478);
			vd = FF(vd, va, vb, vc, x1, s12, 0xe8c7b756);
			vc = FF(vc, vd, va, vb, x2, s13, 0x242070db);
			vb = FF(vb, vc, vd, va, x3, s14, 0xc1bdceee);
			va = FF(va, vb, vc, vd, x4, s11, 0xf57c0faf);
			vd = FF(vd, va, vb, vc, x5, s12, 0x4787c62a);
			vc = FF(vc, vd, va, vb, x6, s13, 0xa8304613);
			vb = FF(vb, vc, vd, va, x7, s14, 0xfd469501);
			va = FF(va, vb, vc, vd, x8, s11, 0x698098d8);
			vd = FF(vd, va, vb, vc, x9, s12, 0x8b44f7af);
			vc = FF(vc, vd, va, vb, x10, s13, 0xffff5bb1);
			vb = FF(vb, vc, vd, va, x11, s14, 0x895cd7be);
			va = FF(va, vb, vc, vd, x12, s11, 0x6b901122);
			vd = FF(vd, va, vb, vc, x13, s12, 0xfd987193);
			vc = FF(vc, vd, va, vb, x14, s13, 0xa679438e);
			vb = FF(vb, vc, vd, va, x15, s14, 0x49b40821);

			/* Round 2 */
			va = GG(va, vb, vc, vd, x1, s21, 0xf61e2562);
			vd = GG(vd, va, vb, vc, x6, s22, 0xc040b340);
			vc = GG(vc, vd, va, vb, x11, s23, 0x265e5a51);
			vb = GG(vb, vc, vd, va, x0, s24, 0xe9b6c7aa);
			va = GG(va, vb, vc, vd, x5, s21, 0xd62f105d);
			vd = GG(vd, va, vb, vc, x10, s22, 0x2441453);
			vc = GG(vc, vd, va, vb, x15, s23, 0xd8a1e681);
			vb = GG(vb, vc, vd, va, x4, s24, 0xe7d3fbc8);
			va = GG(va, vb, vc, vd, x9, s21, 0x21e1cde6);
			vd = GG(vd, va, vb, vc, x14, s22, 0xc33707d6);
			vc = GG(vc, vd, va, vb, x3, s23, 0xf4d50d87);
			vb = GG(vb, vc, vd, va, x8, s24, 0x455a14ed);
			va = GG(va, vb, vc, vd, x13, s21, 0xa9e3e905);
			vd = GG(vd, va, vb, vc, x2, s22, 0xfcefa3f8);
			vc = GG(vc, vd, va, vb, x7, s23, 0x676f02d9);
			vb = GG(vb, vc, vd, va, x12, s24, 0x8d2a4c8a);

			/* Round 3 */
			va = HH(va, vb, vc, vd, x5, s31, 0xfffa3942);
			vd = HH(vd, va, vb, vc, x8, s32, 0x8771f681);
			vc = HH(vc, vd, va, vb, x11, s33, 0x6d9d6122);
			vb = HH(vb, vc, vd, va, x14, s34, 0xfde5380c);
			va = HH(va, vb, vc, vd, x1, s31, 0xa4beea44);
			vd = HH(vd, va, vb, vc, x4, s32, 0x4bdecfa9);
			vc = HH(vc, vd, va, vb, x7, s33, 0xf6bb4b60);
			vb = HH(vb, vc, vd, va, x10, s34, 0xbebfbc70);
			va = HH(va, vb, vc, vd, x13, s31, 0x289b7ec6);
			vd = HH(vd, va, vb, vc, x0, s32, 0xeaa127fa);
			vc = HH(vc, vd, va, vb, x3, s33, 0xd4ef3085);
			vb = HH(vb, vc, vd, va, x6, s34, 0x4881d05);
			va = HH(va, vb, vc, vd, x9, s31, 0xd9d4d039);
			vd = HH(vd, va, vb, vc, x12, s32, 0xe6db99e5);
			vc = HH(vc, vd, va, vb, x15, s33, 0x1fa27cf8);
			vb = HH(vb, vc, vd, va, x2, s34, 0xc4ac5665);

			/* Round 4 */
			va = II(va, vb, vc, vd, x0, s41, 0xf4292244);
			vd = II(vd, va, vb, vc, x7, s42, 0x432aff97);
			vc = II(vc, vd, va, vb, x14, s43, 0xab9423a7);
			vb = II(vb, vc, vd, va, x5, s44, 0xfc93a039);
			va = II(va, vb, vc, vd, x12, s41, 0x655b59c3);
			vd = II(vd, va, vb, vc, x3, s42, 0x8f0ccc92);
			vc = II(vc, vd, va, vb, x10, s43, 0xffeff47d);
			vb = II(vb, vc, vd, va, x1, s44, 0x85845dd1);
			va = II(va, vb, vc, vd, x8, s41, 0x6fa87e4f);
			vd = II(vd, va, vb, vc, x15, s42, 0xfe2ce6e0);
			vc = II(vc, vd, va, vb, x6, s43, 0xa3014314);
			vb = II(vb, vc, vd, va, x13, s44, 0x4e0811a1);
			va = II(va, vb, vc, vd, x4, s41, 0xf7537e82);
			vd = II(vd, va, vb, vc, x11, s42, 0xbd3af235);
			vc = II(vc, vd, va, vb, x2, s43, 0x2ad7d2bb);
			vb = II(vb, vc, vd, va, x9, s44, 0xeb86d391);

			state[2][0] += vgetq_lane_u32(va, 0);
			state[2][1] += vgetq_lane_u32(vb, 0);
			state[2][2] += vgetq_lane_u32(vc, 0);
			state[2][3] += vgetq_lane_u32(vd, 0);
		}
	}
	if (n_blocks4 > min_n_blocks)
	{
		for (int i = min_n_blocks; i < n_blocks4; i += 1)
		{
			for (int i1 = 0; i1 < 16; ++i1)
			{
				x[3][i1] = (paddedMessage4[4 * i1 + i * 64]) |
					(paddedMessage4[4 * i1 + 1 + i * 64] << 8) |
					(paddedMessage4[4 * i1 + 2 + i * 64] << 16) |
					(paddedMessage4[4 * i1 + 3 + i * 64] << 24);
			}
			uint32x4_t va = vdupq_n_u32(state[3][0]);
			uint32x4_t vb = vdupq_n_u32(state[3][1]);
			uint32x4_t vc = vdupq_n_u32(state[3][2]);
			uint32x4_t vd = vdupq_n_u32(state[3][3]);
			uint32x4_t x0 = vdupq_n_u32(x[3][0]);
			uint32x4_t x1 = vdupq_n_u32(x[3][1]);
			uint32x4_t x2 = vdupq_n_u32(x[3][2]);
			uint32x4_t x3 = vdupq_n_u32(x[3][3]);
			uint32x4_t x4 = vdupq_n_u32(x[3][4]);
			uint32x4_t x5 = vdupq_n_u32(x[3][5]);
			uint32x4_t x6 = vdupq_n_u32(x[3][6]);
			uint32x4_t x7 = vdupq_n_u32(x[3][7]);
			uint32x4_t x8 = vdupq_n_u32(x[3][8]);
			uint32x4_t x9 = vdupq_n_u32(x[3][9]);
			uint32x4_t x10 = vdupq_n_u32(x[3][10]);
			uint32x4_t x11 = vdupq_n_u32(x[3][11]);
			uint32x4_t x12 = vdupq_n_u32(x[3][12]);
			uint32x4_t x13 = vdupq_n_u32(x[3][13]);
			uint32x4_t x14 = vdupq_n_u32(x[3][14]);
			uint32x4_t x15 = vdupq_n_u32(x[3][15]);
			va = FF(va, vb, vc, vd, x0, s11, 0xd76aa478);
			vd = FF(vd, va, vb, vc, x1, s12, 0xe8c7b756);
			vc = FF(vc, vd, va, vb, x2, s13, 0x242070db);
			vb = FF(vb, vc, vd, va, x3, s14, 0xc1bdceee);
			va = FF(va, vb, vc, vd, x4, s11, 0xf57c0faf);
			vd = FF(vd, va, vb, vc, x5, s12, 0x4787c62a);
			vc = FF(vc, vd, va, vb, x6, s13, 0xa8304613);
			vb = FF(vb, vc, vd, va, x7, s14, 0xfd469501);
			va = FF(va, vb, vc, vd, x8, s11, 0x698098d8);
			vd = FF(vd, va, vb, vc, x9, s12, 0x8b44f7af);
			vc = FF(vc, vd, va, vb, x10, s13, 0xffff5bb1);
			vb = FF(vb, vc, vd, va, x11, s14, 0x895cd7be);
			va = FF(va, vb, vc, vd, x12, s11, 0x6b901122);
			vd = FF(vd, va, vb, vc, x13, s12, 0xfd987193);
			vc = FF(vc, vd, va, vb, x14, s13, 0xa679438e);
			vb = FF(vb, vc, vd, va, x15, s14, 0x49b40821);

			/* Round 2 */
			va = GG(va, vb, vc, vd, x1, s21, 0xf61e2562);
			vd = GG(vd, va, vb, vc, x6, s22, 0xc040b340);
			vc = GG(vc, vd, va, vb, x11, s23, 0x265e5a51);
			vb = GG(vb, vc, vd, va, x0, s24, 0xe9b6c7aa);
			va = GG(va, vb, vc, vd, x5, s21, 0xd62f105d);
			vd = GG(vd, va, vb, vc, x10, s22, 0x2441453);
			vc = GG(vc, vd, va, vb, x15, s23, 0xd8a1e681);
			vb = GG(vb, vc, vd, va, x4, s24, 0xe7d3fbc8);
			va = GG(va, vb, vc, vd, x9, s21, 0x21e1cde6);
			vd = GG(vd, va, vb, vc, x14, s22, 0xc33707d6);
			vc = GG(vc, vd, va, vb, x3, s23, 0xf4d50d87);
			vb = GG(vb, vc, vd, va, x8, s24, 0x455a14ed);
			va = GG(va, vb, vc, vd, x13, s21, 0xa9e3e905);
			vd = GG(vd, va, vb, vc, x2, s22, 0xfcefa3f8);
			vc = GG(vc, vd, va, vb, x7, s23, 0x676f02d9);
			vb = GG(vb, vc, vd, va, x12, s24, 0x8d2a4c8a);

			/* Round 3 */
			va = HH(va, vb, vc, vd, x5, s31, 0xfffa3942);
			vd = HH(vd, va, vb, vc, x8, s32, 0x8771f681);
			vc = HH(vc, vd, va, vb, x11, s33, 0x6d9d6122);
			vb = HH(vb, vc, vd, va, x14, s34, 0xfde5380c);
			va = HH(va, vb, vc, vd, x1, s31, 0xa4beea44);
			vd = HH(vd, va, vb, vc, x4, s32, 0x4bdecfa9);
			vc = HH(vc, vd, va, vb, x7, s33, 0xf6bb4b60);
			vb = HH(vb, vc, vd, va, x10, s34, 0xbebfbc70);
			va = HH(va, vb, vc, vd, x13, s31, 0x289b7ec6);
			vd = HH(vd, va, vb, vc, x0, s32, 0xeaa127fa);
			vc = HH(vc, vd, va, vb, x3, s33, 0xd4ef3085);
			vb = HH(vb, vc, vd, va, x6, s34, 0x4881d05);
			va = HH(va, vb, vc, vd, x9, s31, 0xd9d4d039);
			vd = HH(vd, va, vb, vc, x12, s32, 0xe6db99e5);
			vc = HH(vc, vd, va, vb, x15, s33, 0x1fa27cf8);
			vb = HH(vb, vc, vd, va, x2, s34, 0xc4ac5665);

			/* Round 4 */
			va = II(va, vb, vc, vd, x0, s41, 0xf4292244);
			vd = II(vd, va, vb, vc, x7, s42, 0x432aff97);
			vc = II(vc, vd, va, vb, x14, s43, 0xab9423a7);
			vb = II(vb, vc, vd, va, x5, s44, 0xfc93a039);
			va = II(va, vb, vc, vd, x12, s41, 0x655b59c3);
			vd = II(vd, va, vb, vc, x3, s42, 0x8f0ccc92);
			vc = II(vc, vd, va, vb, x10, s43, 0xffeff47d);
			vb = II(vb, vc, vd, va, x1, s44, 0x85845dd1);
			va = II(va, vb, vc, vd, x8, s41, 0x6fa87e4f);
			vd = II(vd, va, vb, vc, x15, s42, 0xfe2ce6e0);
			vc = II(vc, vd, va, vb, x6, s43, 0xa3014314);
			vb = II(vb, vc, vd, va, x13, s44, 0x4e0811a1);
			va = II(va, vb, vc, vd, x4, s41, 0xf7537e82);
			vd = II(vd, va, vb, vc, x11, s42, 0xbd3af235);
			vc = II(vc, vd, va, vb, x2, s43, 0x2ad7d2bb);
			vb = II(vb, vc, vd, va, x9, s44, 0xeb86d391);

			state[3][0] += vgetq_lane_u32(va, 0);
			state[3][1] += vgetq_lane_u32(vb, 0);
			state[3][2] += vgetq_lane_u32(vc, 0);
			state[3][3] += vgetq_lane_u32(vd, 0);
		}
	}

	// 下面的处理，在理解上较为复杂
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[0][i];
		state[0][i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
			((value & 0xff00) << 8) |	 // 将次低字节左移
			((value & 0xff0000) >> 8) |	 // 将次高字节右移
			((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[1][i];
		state[1][i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
			((value & 0xff00) << 8) |	 // 将次低字节左移
			((value & 0xff0000) >> 8) |	 // 将次高字节右移
			((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[2][i];
		state[2][i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
			((value & 0xff00) << 8) |	 // 将次低字节左移
			((value & 0xff0000) >> 8) |	 // 将次高字节右移
			((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[3][i];
		state[3][i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
			((value & 0xff00) << 8) |	 // 将次低字节左移
			((value & 0xff0000) >> 8) |	 // 将次高字节右移
			((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}
	// 输出最终的hash结果
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	delete[] paddedMessage1;
	delete[] paddedMessage2;
	delete[] paddedMessage3;
	delete[] paddedMessage4;
	delete[] messageLength1;
	delete[] messageLength2;
	delete[] messageLength3;
	delete[] messageLength4;
}