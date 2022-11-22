import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
import numpy as np
import cv2
import time


def mixing_gradient(p_start_y, p_start_x, p_mask, p_tgt, p_src):
    mask_row, mask_col = p_mask.shape[0], p_mask.shape[1]
    tgt_row, tgt_col = p_tgt.shape[0], p_tgt.shape[1]

    if p_start_y + mask_row > tgt_row or p_start_x + mask_col > tgt_col:
        print('Out of range!')
        return p_tgt
    else:
        p_src = np.array(p_src, np.float32)
        p_tgt = np.array(p_tgt, np.float32)

        new_mask = np.zeros([tgt_row, tgt_col], np.float32)
        new_src = np.zeros([tgt_row, tgt_col, 3], np.float32)
        new_mask[p_start_y:p_start_y + mask_row, p_start_x:p_start_x + mask_col] = p_mask
        new_src[p_start_y:p_start_y + mask_row, p_start_x:p_start_x + mask_col] = p_src

        pix_y, pix_x = np.where(new_mask)
        pix_old_y, pix_old_x = np.where(p_mask)

        pix_sequence = np.zeros([tgt_row, tgt_col])
        pix_sequence[pix_y, pix_x] = np.arange(pix_y.shape[0])

        direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        anc_list = []
        edge_list = []

        lap_matrix = sparse.lil_matrix((pix_y.shape[0], pix_y.shape[0]), dtype=np.float32)

        for idx in range(pix_y.shape[0]):
            row = pix_y[idx]
            col = pix_x[idx]
            old_row = pix_old_y[idx]
            old_col = pix_old_x[idx]

            anc_dir = int()

            tmp_anc = np.zeros([3], np.float32)
            tmp_edge = np.zeros([3], np.float32)

            for direct in direction:
                if 0 <= old_row + direct[0] < mask_row and 0 <= old_col + direct[1] < mask_col:
                    anc_dir += 1

                    # 获取源图像和目标图像锚点和该方向上点的像素值
                    src_dir = p_src[old_row + direct[0], old_col + direct[1], :]
                    tgt_dir = p_tgt[row + direct[0], col + direct[1], :]
                    src_anc = p_src[old_row, old_col, :]
                    tgt_anc = p_tgt[row, col, :]

                    # 比较各方向梯度绝对值（三通道平方和开根），取较大的值
                    if np.linalg.norm(src_anc - src_dir) > np.linalg.norm(tgt_anc - tgt_dir):
                        tmp_anc += (src_anc - src_dir)
                    else:
                        tmp_anc += (tgt_anc - tgt_dir)

                    if p_mask[old_row + direct[0], old_col + direct[1]] == 0:
                        tmp_edge += p_tgt[row + direct[0], col + direct[1], :]
                    else:
                        lap_matrix[idx, pix_sequence[row + direct[0], col + direct[1]]] = -1

            lap_matrix[idx, idx] = anc_dir

            anc_list.append(tmp_anc)

            edge_list.append(tmp_edge)

        anc_list = np.array(anc_list, np.float32)
        edge_list = np.array(edge_list, np.float32)

        left = lap_matrix.tocoo()
        b = anc_list + edge_list

        res_b = splinalg.cg(left, b[:, 0])[0]
        res_g = splinalg.cg(left, b[:, 1])[0]
        res_r = splinalg.cg(left, b[:, 2])[0]

        res = p_tgt.copy()
        res[pix_y, pix_x] = np.clip(np.column_stack((res_b, res_g, res_r)), 0, 255)
        res = np.array(res, np.uint8)

        return res


if __name__ == "__main__":
    start_time = time.time()
    src = cv2.imread('./examples/2.jpg')
    tgt = cv2.imread('./examples/1.jpg')
    mask = cv2.imread('./examples/mask.jpg')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    tgt = mixing_gradient(10, 20, mask, tgt, src)

    end_time = time.time()
    print(end_time - start_time)

    cv2.imshow('tgt', tgt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
