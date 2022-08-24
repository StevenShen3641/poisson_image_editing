import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
import numpy as np
import cv2
import time


def poisson_image_editing(p_start_y, p_start_x, p_mask, p_tgt, p_src):
    """
    泊松图像编辑函数
    :param p_start_y: 源图像左上角像素的坐标y值
    :param p_start_x: 源图像左上角像素的坐标x值
    :param p_mask: 掩膜图像，必须是双通道的，大小和源图像相同
    :param p_tgt: 目标图像
    :param p_src: 源图像，必须比目标图像小或相等
    :return:
    """
    mask_row, mask_col = p_mask.shape[0], p_mask.shape[1]  # 掩膜的宽度和长度
    tgt_row, tgt_col = p_tgt.shape[0], p_tgt.shape[1]  # 目标图像的宽度和长度

    # 判断：源图像映射在目标图像上是否超出了目标图像边界
    if p_start_y + mask_row > tgt_row or p_start_x + mask_col > tgt_col:
        # 提示并返回目标图像
        print('Out of range!')
        return p_tgt
    else:
        p_src = np.array(p_src, np.float32)  # 位深改为浮点数，防止np.uint8超出255后归零，下同
        p_tgt = np.array(p_tgt, np.float32)

        # 按源图像左上角像素的坐标重建和目标图像大小相同的掩膜和源图像，注意掩膜的通道数为2
        new_mask = np.zeros([tgt_row, tgt_col], np.float32)
        new_src = np.zeros([tgt_row, tgt_col, 3], np.float32)
        new_mask[p_start_y:p_start_y + mask_row, p_start_x:p_start_x + mask_col] = p_mask
        new_src[p_start_y:p_start_y + mask_row, p_start_x:p_start_x + mask_col] = p_src

        # 获取新掩膜和原掩膜每个锚点的坐标，纵坐标存于pix_y，横坐标存于pix_x，一一对应
        pix_y, pix_x = np.where(new_mask)
        pix_old_y, pix_old_x = np.where(p_mask)

        pix_sequence = np.zeros([tgt_row, tgt_col])  # 创建和目标图像大小相同的全黑模板
        pix_sequence[pix_y, pix_x] = np.arange(pix_y.shape[0])  # 模板上每个掩膜的锚点按从上到下，从左到右的顺序依次标上序号

        direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # 拉普拉斯算子的四个方向

        anc_list = []  # 每个掩膜锚点的拉普拉斯值（三通道）
        edge_list = []  # 若像素为边界点，则此列表存放目标图像的边界像素值（三通道）

        # 为了节省内存空间，采用稀疏矩阵存放线性方程组左侧值
        lap_matrix = sparse.lil_matrix((pix_y.shape[0], pix_y.shape[0]), dtype=np.float32)

        for idx in range(pix_y.shape[0]):
            # 获取掩膜锚点在目标和源图像上的坐标
            row = pix_y[idx]
            col = pix_x[idx]
            old_row = pix_old_y[idx]
            old_col = pix_old_x[idx]

            anc_dir = int()  # 拉普拉斯算子方向数

            tmp_anc = np.zeros([3], np.float32)  # 临时存放拉普拉斯值
            tmp_edge = np.zeros([3], np.float32)  # 临时存放边界像素值

            # 对各个方向遍历
            for direct in direction:
                # 判断该方向是否超出源图像范围
                if 0 <= old_row + direct[0] < mask_row and 0 <= old_col + direct[1] < mask_col:
                    anc_dir += 1  # 方向数加一
                    tmp_anc -= p_src[old_row + direct[0], old_col + direct[1], :]  # 减去该方向像素值

                    # 判断是否为边界点
                    if p_mask[old_row + direct[0], old_col + direct[1]] == 0:

                        tmp_edge += p_tgt[row + direct[0], col + direct[1], :]  # 存放边界值
                    else:
                        # 找出pix_sequence该方向对应的下标，并在稀疏矩阵中标出
                        lap_matrix[idx, pix_sequence[row + direct[0], col + direct[1]]] = -1

            tmp_anc += (p_src[old_row, old_col, :] * anc_dir)  # 加上锚点对应源图像上的像素值乘方向数

            lap_matrix[idx, idx] = anc_dir  # 标出对角线上锚点被加次数

            # 存放至算子集和边界集
            anc_list.append(tmp_anc)
            edge_list.append(tmp_edge)

        # 将列表转换为ndarray，同样注意位深
        anc_list = np.array(anc_list, np.float32)
        edge_list = np.array(edge_list, np.float32)

        left = lap_matrix.tocoo()  # 将稀疏矩阵转化为坐标形式，减小高斯消元的时间复杂度
        b = anc_list + edge_list  # 方程右侧值

        # 高斯消元
        res_b = splinalg.cg(left, b[:, 0])[0]
        res_g = splinalg.cg(left, b[:, 1])[0]
        res_r = splinalg.cg(left, b[:, 2])[0]

        # 将计算值附在目标图像上
        res = p_tgt.copy()
        res[pix_y, pix_x] = np.clip(np.column_stack((res_b, res_g, res_r)), 0, 255)
        res = np.array(res, np.uint8)

        return res


if __name__ == "__main__":
    start_time = time.time()

    # 导入图像
    src1 = cv2.imread('./src1.jpg')
    src2 = cv2.imread('./src2.jpg')
    src3 = cv2.imread('./src3.jpg')
    tgt = cv2.imread('./tgt.jpg')

    # 掩膜转化为二通道灰度图像
    mask1 = cv2.imread('./mask1.jpg')
    mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.imread('./mask2.jpg')
    mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
    mask3 = cv2.imread('./mask3.jpg')
    mask3 = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
    tgt = poisson_image_editing(120, 20, mask1, tgt, src1)
    tgt = poisson_image_editing(120, 150, mask2, tgt, src2)
    tgt = poisson_image_editing(10, 40, mask3, tgt, src3)

    end_time = time.time()
    print('It spends {time} seconds.'.format(time=end_time - start_time))

    # 展示
    cv2.imshow('tgt', tgt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
