#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>

// 打印 int32x4x2_t 的内容
void print_int32x4x2(const char* name, int32x4x2_t data) {
    printf("%-10s: [", name);
    for (int i = 0; i < 2; i++) {
        int32_t lane0 = vgetq_lane_s32(data.val[i], 0);
        int32_t lane1 = vgetq_lane_s32(data.val[i], 1);
        int32_t lane2 = vgetq_lane_s32(data.val[i], 2);
        int32_t lane3 = vgetq_lane_s32(data.val[i], 3);
        
        printf("[%2d, %2d, %2d, %2d]", lane0, lane1, lane2, lane3);
        if (i == 0) printf(", ");
    }
    printf("]\n");
}

int main() {
    // 1. 创建测试数据 (8个32位整数)
    int32_t src_a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t src_b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    int32_t dst[8];
    
    // 2. 加载数据到 int32x4x2_t
    int32x4x2_t a = vld1q_s32_x2(src_a);
    int32x4x2_t b = vld1q_s32_x2(src_b);
    
    // 3. 打印加载的数据
    printf("---- 加载数据 ----\n");
    print_int32x4x2("a", a);
    print_int32x4x2("b", b);
    
    // 4. 执行向量加法 (8个元素同时相加)
    int32x4x2_t sum;
    sum.val[0] = vaddq_s32(a.val[0], b.val[0]);  // 前4个元素相加
    sum.val[1] = vaddq_s32(a.val[1], b.val[1]);  // 后4个元素相加
    
    // 5. 打印加法结果
    printf("\n---- 加法结果 ----\n");
    print_int32x4x2("sum", sum);
    
    // 6. 存储结果到内存
    vst1q_s32_x2(dst, sum);
    
    // 7. 打印内存中的结果
    printf("\n---- 存储结果 ----\n");
    printf("dst: [");
    for (int i = 0; i < 8; i++) {
        printf("%2d", dst[i]);
        if (i < 7) printf(", ");
    }
    printf("]\n");
    
    // 8. 验证结果
    int valid = 1;
    for (int i = 0; i < 8; i++) {
        if (dst[i] != src_a[i] + src_b[i]) {
            valid = 0;
            break;
        }
    }
    printf("\n验证: %s\n", valid ? "通过" : "失败s");
    
    return 0;
}