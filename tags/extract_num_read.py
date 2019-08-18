import numpy as np
import logging
from sklearn.cluster import KMeans
import copy

def bbox_to_centr_coord(bboxs:np.ndarray)->np.ndarray:
    # 计算bbox的中心坐标
    centr_x, centr_y = (bboxs[:, 0]+bboxs[:, 2])/2, (bboxs[:, 1]+bboxs[:, 3])/2
    return centr_x, centr_y

def sort_at_xcoord(bboxs:np.ndarray, category:np.ndarray)->np.ndarray:
    # 按照x坐标进行排序
    centr_x,centr_y = bbox_to_centr_coord(bboxs)
    space_idx = np.argsort(centr_x)
    sort_centr_coord = np.stack([centr_x[space_idx],centr_y[space_idx]],axis=1)
    sort_category = category[space_idx]
    return sort_centr_coord,sort_category,bboxs[space_idx]


def compute_space(lst_sort:np.ndarray)->np.ndarray:
    # 计算已排序列表相邻两个数字之间的差值
    space = []
    for i in range(len(lst_sort)-1):
        space.append(lst_sort[i+1]-lst_sort[i])
    return np.array(space)


def compute_divide(sort_centr_coord:np.ndarray)->np.ndarray:
    # 计算刻度,返回划分点np数组
    space = compute_space(sort_centr_coord[:, 0])  # 相邻点之间的差值 n-1
    # logging.debug(space)
    # space = _space.copy()
    # # space = np.delete(_space,np.argmax(_space),axis=0)
    # sort_idx = np.argsort(_space)
    # if space[sort_idx[-1]]/space[sort_idx[-2]] >= 1.5:  # 删除异常数值
    #     logging.debug(space)
    #     space = np.delete(space, sort_idx[-1], axis=0)
    # threshold = space.mean()#(space.max()+space.min())/2  # 阈值
    # threshold = KMeans(4).fit(space.reshape(-1,1)).cluster_centers_.mean()
    threshold = 28  # 经验值 默认图片300像素
    print('space {}'.format(space))
    print('threshold {}'.format(threshold))
    logging.debug(space)
    # logging.debug(KMeans(1).fit(space.reshape(-1,1)).cluster_centers_.mean())
    # logging.debug(KMeans(2).fit(space.reshape(-1,1)).cluster_centers_.mean())
    # logging.debug(KMeans(3).fit(space.reshape(-1, 1)).cluster_centers_.mean())
    # logging.debug(KMeans(4).fit(space.reshape(-1, 1)).cluster_centers_.mean())
    # logging.debug(KMeans(5).fit(space.reshape(-1, 1)).cluster_centers_.mean())
    divide = (space < threshold)  # 划分 n-1
    divide_result = []  # 分界点列表
    for i in range(len(divide)):  # 0 to n-1
        if divide[i]:  # 低于阈值 视为同一个数
            continue
        else:
            divide_result.append(i+1)  # 高于阈值视为分界点 +1是为还原位置
    return np.array(divide_result)


def compute_graduation(divide:np.ndarray,sort_center_coord:np.ndarray,sort_category:np.ndarray)->np.ndarray:
    # 1D 2D 1D
    '''根据分界点，计算实际刻度(数值和center)'''
    graduation_center = []
    graduation_class = []
    # 第一组
    graduation_center.append(sort_center_coord[0:divide[0]])
    graduation_class.append(sort_category[0:divide[0]])
    # 中间组
    left_i = 0
    for i in range(1,len(divide)):
        graduation_center.append(sort_center_coord[divide[left_i]:divide[i]])
        graduation_class.append(sort_category[divide[left_i]:divide[i]])
        left_i = i
    # 最后一组
    graduation_center.append(sort_center_coord[divide[-1]:])
    graduation_class.append(sort_category[divide[-1]:])
    # 刻度融合
    result_class = [int(''.join(i)) for i in graduation_class]
    result_center = [[i[:, 0].mean(), i[:, 1].mean()] for i in graduation_center]  # 融合中心
    # result_center = [[(i[:,[0,2]].min()+i[:,[0,2]].max())/2, (i[:,[1,3]].min()+i[:,[1,3]].max())/2] for i in graduation_center]  # 融合边界框
    return np.array(result_center), np.array(result_class)


def compute_num_read(center_coord_x: np.ndarray, _class: np.ndarray,center_finger_x,post):
    # 最终计算读数 1D 1D 1D
    # 距离排序
    distance_sort_id = np.argsort(np.absolute(center_finger_x - center_coord_x))
    # 计算像素和实际距离比例
    _class_copy = copy.deepcopy(_class)
    if ''.join(_class_copy.astype(str)).find('991')!=-1 and 99 in _class_copy and 1 in _class_copy:  # 处理指针在99和1中间的情况
        _class_copy[np.argwhere(_class_copy==99).squeeze()+1:]+=100
    ratio = (compute_space(_class_copy)/compute_space(center_coord_x)).mean()  # 计算比例系数,平均
    # ratio = compute_space(_class[distance_sort_id[0:2]])/compute_space(center_coord_x[0:2])  # 计算比例系数，最近
    # 找距离最近得两个计算得数
    if len(distance_sort_id)>1:
        num_read1 = (center_finger_x - center_coord_x[distance_sort_id[0]])*ratio+_class[distance_sort_id[0]]
        num_read2 = (center_finger_x - center_coord_x[distance_sort_id[1]])*ratio+_class[distance_sort_id[1]]
        if post:
            num_read = (post_num_read(num_read1)+post_num_read(num_read2))/2
        else:
            num_read = (num_read1 + num_read2) / 2
    else:
        num_read = (center_finger_x - center_coord_x[distance_sort_id[0]])*ratio+_class[distance_sort_id[0]]
        if post:
            num_read = post_num_read(num_read)
    return num_read  # 返回平均


def detect_m(sort_category):
    '''判断是否存在整米'''
    str123='123'
    str99='9899'
    str_detect = ''.join(sort_category) # 待检测字符串
    flag123= str_detect.find(str123)
    flag99 = str_detect.find(str99)
    # 情况1 存在99 存在123 保留整米刻度
    if flag123!=-1 and flag99!=-1:
        logging.info('情况1 整米在123和99之间保留整米刻度，但是合并数值')
        flag = 1
    # 情况2 存在99 不存在123 删除整米刻度
    elif flag123==-1 and flag99!=-1:
        logging.info('情况2 整米在99右侧删除整米刻度')
        flag=2
    # 情况3 不存在99 存在123 删除整米刻度
    elif flag123!=-1 and flag99==-1:
        logging.info('情况3 整米在123左侧删除整米刻度')
        flag=3
    # 情况4 都不存在 不存在整米刻度
    else:
        logging.info('情况4 不存在整米刻度,不做处理')
        flag=4
    return flag,flag99,flag123


def delete_m(sort_centr_coord, sort_category,sort_bbox,idx1,idx2,meth='delete'):
    if meth =='merger':
        sort_category[idx1] = '0'
        sort_category = np.delete(sort_category,idx2,axis=0)
        # 合并中心坐标,留1删1
        sort_centr_coord[idx1] = (sort_centr_coord[idx1] + sort_centr_coord[idx2]) / 2
        sort_centr_coord = np.delete(sort_centr_coord, idx2, axis=0)
        # 合并box，留1删1
        sort_bbox[idx1] = (sort_bbox[idx1] + sort_bbox[idx2]) / 2
        sort_bbox = np.delete(sort_bbox, [idx2], axis=0)
    else:
        sort_category = np.delete(sort_category, [idx1, idx2], axis=0)
        sort_centr_coord = np.delete(sort_centr_coord, [idx1, idx2], axis=0)
        sort_bbox = np.delete(sort_bbox, [idx1, idx2], axis=0)
    return sort_centr_coord, sort_category,sort_bbox


def press_m(press_flag,sort_centr_coord, sort_category,sort_bbox):
    '''处理整米情况'''
    if press_flag[0] == 1:  # 情况1
        if (press_flag[2]-press_flag[1]-1) == 2:
            logging.info('0-9m 不用合并')
        else:
            logging.info('>10m  合并')
            # 合并类,留1删1
            sort_centr_coord, sort_category, sort_bbox = delete_m(sort_centr_coord, sort_category,sort_bbox,press_flag[2]-1,press_flag[2]-2)
            # sort_category[press_flag[2]-2] = '0'
            # sort_category = np.delete(sort_category,[press_flag[2]-1],axis=0)
            # # 合并中心坐标,留1删1
            # sort_centr_coord[press_flag[2]-2] = (sort_centr_coord[press_flag[2]-1]+sort_centr_coord[press_flag[2]-2])/2
            # sort_centr_coord = np.delete(sort_centr_coord,[press_flag[2]-1],axis=0)
            # # 合并box，留1删1
            # sort_bbox[press_flag[2] - 2] = (sort_bbox[press_flag[2] - 1] + sort_bbox[press_flag[2] - 2]) / 2
            # sort_bbox = np.delete(sort_bbox, [press_flag[2] - 1], axis=0)
    elif press_flag[0] == 2:
        # 删除99之后得类及坐标
        logging.info('删除9899之后的类及坐标')
        if len(sort_category)-(press_flag[1]+4)>2:   # 保留 1和2
            # 合并类,留1删1
            sort_centr_coord, sort_category, sort_bbox = delete_m(sort_centr_coord, sort_category, sort_bbox,
                                                                  press_flag[1]+4, press_flag[1]+5)
            # sort_category[press_flag[1]+4] = '0'
            # sort_category = np.delete(sort_category, [press_flag[1]+5], axis=0)
            # # sort_category = np.delete(sort_category,[press_flag[1]+4,press_flag[1]+5],axis=0)
            # sort_centr_coord = np.delete(sort_centr_coord,[press_flag[1]+4,press_flag[1]+5],axis=0)
            # sort_bbox = np.delete(sort_bbox,[press_flag[1]+4,press_flag[1]+5],axis=0)
        else:  # 不存在 1 2 则全部删除
            sort_category=sort_category[:(press_flag[1]+4)]
            sort_centr_coord=sort_centr_coord[:(press_flag[1]+4)]
            sort_bbox=sort_bbox[:(press_flag[1]+4)]
    elif press_flag[0] == 3:
        logging.info('删除123之前的类及坐标')
        if press_flag[2]>1:
            sort_centr_coord, sort_category, sort_bbox = delete_m(sort_centr_coord, sort_category, sort_bbox,
                                                                  press_flag[2]-1, press_flag[2]-2)
            # sort_category = np.delete(sort_category, [press_flag[2]-1, press_flag[2]-2], axis=0)
            # sort_centr_coord = np.delete(sort_centr_coord, [press_flag[2]-1, press_flag[2]-2], axis=0)
            # sort_bbox = np.delete(sort_bbox, [press_flag[2]-1, press_flag[2]-2], axis=0)
        else:
            sort_category = sort_category[press_flag[2]:]
            sort_centr_coord = sort_centr_coord[press_flag[2]:]
            sort_bbox = sort_bbox[press_flag[2]:]
    else:
        logging.info('不存在整米刻度,不做处理')
    return sort_centr_coord,sort_category,sort_bbox


def detect_single(sort_category):
    '''探测1-9刻度是否存在'''
    logging.info('探测1-9刻度是否存在')
    count = 0  # 计数
    max_count = 0
    # 检测是否存在
    for i in range(len(sort_category)-1):
        if (int(sort_category[i+1])-int(sort_category[i]))==1:
            count += 1
        else:
            count = 0
        if count > max_count:
            max_count = count
    # 返回处理策略
    if max_count > 1:
        logging.info('存在1-9刻度 {}'.format(max_count))
        flag = True
    else:
        flag = False
        logging.info('不存在1-9刻度 {}'.format(max_count))
    return flag
        # if '1' in sort_category:
        #     idx = np.argwhere(np.array(sort_category)=='1').squeeze()
        #     logging.info('存在整米刻度,1刻度位置{}'.format(idx))
        #     if idx == 2:
        #         logging.info('合并前两个')
        #         return 2
        #     elif idx == 1:
        #         logging.info('删掉前一个')
        #         return 1


def judgement_only_single(sort_category):
    '''处理1-9特殊刻度'''
    logging.info('判断是否只存在0-9刻度')
    flag = False
    str_detect = ''.join(sort_category)  # 待检测字符串
    flag99 = str_detect.find('99')
    flag10 = str_detect.find('10')
    if flag99==flag10==-1:
        logging.info("只存在0-9刻度")
        flag = True
    else:
        logging.info("不只存在0-9刻度或不存在0-9刻度")
    return flag


def delete_single(result_center,result_class):
    '''删除单数坐标'''
    logging.info("去除单数坐标")
    th = 10
    if 10 in result_class.astype(int):
        logging.info("去除9右侧的单数")
        idx10 = np.argwhere(result_class==10).squeeze()
        logging.info("10的索引{}".format(idx10))
        del_idx = np.argwhere(result_class[idx10:]<th).squeeze()+idx10
    elif 99 in result_class.astype(int):
        logging.info("去除1左侧的单数")
        idx99 = np.argwhere(result_class == 99).squeeze()
        logging.info("99的索引{}".format(idx99))
        del_idx = np.argwhere(result_class[:idx99] < th).squeeze()
    else:
        logging.info("去除一般情况下的单数")
        del_idx = np.argwhere(result_class < th).squeeze()
    logging.info("应删除的索引{}".format(del_idx))
    result_center = np.delete(result_center, del_idx, axis=0)
    result_class = np.delete(result_class, del_idx, axis=0)
    return result_center,result_class


def delete_three(result_center,result_class):
    '''删除两位以上的数'''
    logging.info("去除三位数坐标")
    th = 99
    del_idx = np.argwhere(result_class > th).squeeze()
    result_center = np.delete(result_center, del_idx, axis=0)
    result_class = np.delete(result_class, del_idx, axis=0)
    logging.info("去除三位数坐标后{} {}".format(result_center,result_class))
    return result_center,result_class


def keep_continue(result_center, result_class):
    logging.info("探测保留连续数字，删除非连续数字")
    count = 0  # 计数
    count_lst = []
    pointer_lst = []
    # 检测是否存在
    for i in range(len(result_class)-1):
        if (int(result_class[i+1]) - int(result_class[i])) == 1:
            count += 1
            if i == (len(result_class)-2):
                count_lst.append(count)
                pointer_lst.append(i)
        else:
            count_lst.append(count)
            pointer_lst.append(i-1)
            count = 0
    count_lst,pointer_lst = np.array(count_lst),np.array(pointer_lst)
    logging.debug('count_lst{} pointer_lst{}'.format(count_lst,pointer_lst))
    pointer_lst = pointer_lst[count_lst > 0]  # 仅保留连续数字
    count_lst = count_lst[count_lst>0]
    keep = []
    for i in range(len(count_lst)):
        keep.extend(range(pointer_lst[i]+1-count_lst[i], pointer_lst[i]+2))
    logging.debug('keep {} keep_class{}'.format(keep,result_class[keep]))
    return result_center[keep], result_class[keep]


def num_read_main(bboxs,category,finger_bbox):
    '''计算读数主函数
    bboxs np (N,4)
    category np (N,)
    finger_bbox np (N,1)
    '''
    logging.info("*"*10+"读数计算程序"+"*"*10)
    logging.info('原始数据\nclass size:{}==>{}\nboxes size:{} boxes==>\n{}'.format(len(category),category, len(bboxs), bboxs))
    sort_centr_coord, sort_category,sort_bbox = sort_at_xcoord(bboxs,category)
    logging.info('1 排序后结果\n{}\n{}\n{}'.format(sort_centr_coord, sort_category, sort_bbox))
    print('1 排序后结果\n{}'.format(sort_category))
    post = True
    center_finger = np.concatenate(bbox_to_centr_coord(finger_bbox[np.newaxis, ...]))
    if center_finger[0] < sort_centr_coord[0, 0]:
        # 指针左侧无读数：
        post = False
    # logging.info('2 数据清洗,主要是处理整米刻度')
    # press_flag = detect_m(sort_category)
    # logging.debug(press_flag)
    # sort_centr_coord, sort_category, sort_bbox=press_m(press_flag,sort_centr_coord, sort_category,sort_bbox)
    # logging.info('3 预处理后的干净数据\n{}\n{}\n{}'.format(sort_centr_coord, sort_category, sort_bbox))
    if detect_single(sort_category) and judgement_only_single(sort_category):
            logging.info("4 只存在0-9刻度,则无需划分刻度")
            result_center, result_class = sort_centr_coord,sort_category.astype(int)
            # 删除右侧边界 1
            # logging.debug(result_class<10)
            # keep = np.argwhere(result_class<10).squeeze()
            # logging.info("保留索引{}".format(keep))
            # result_center, result_class = keep_continue(result_center, result_class)
            # logging.info(' 保留连续的刻度和坐标\n刻度:\n{}\n坐标:\n{}'.format(result_class, result_center))
            # if result_class[-1] == 1:00
    else:
        logging.info('4 存在>9刻度,正常划分刻度')
        divide = compute_divide(sort_centr_coord)
        result_center, result_class=compute_graduation(divide, sort_centr_coord, sort_category)
        logging.info(' 划分后的刻度和坐标\n刻度:\n{}\n坐标:\n{}'.format(result_class, result_center))
        print(' 划分后的刻度和坐标\n刻度:\n{}'.format(result_class))
        result_center, result_class=delete_single(*delete_three(result_center, result_class))
        logging.info(' 删除多余得单or三位数刻度和坐标\n刻度:\n{}\n坐标:\n{}'.format(result_class, result_center))
    result_center, result_class=keep_continue(result_center, result_class)
    logging.info(' 保留连续的刻度和坐标\n刻度:\n{}\n坐标:\n{}'.format(result_class, result_center))
    logging.info('5 获取实际刻度和坐标\n实际刻度:\n{}\n坐标:\n{}'.format(result_class, result_center))
    print('实际刻度：{}'.format(result_class))
    if result_class.size == 0:
        return None
    # if finger_bbox is None:
    #     finger_idx = np.random.randint(low=0,high=len(result_class))
    #     center_finger = result_center[finger_idx]
    #     print('仅仅用于测试目的,{}充当指针{}'.format(result_class[finger_idx],result_center[finger_idx]))
    # else:

    logging.info('6 指针中心坐标\n{}'.format(center_finger))
    num_read = compute_num_read(result_center[:,0],result_class,center_finger[0],post)
    return num_read


def post_num_read(num_read):
    '''处理读数越界'''
    num_read%=100
    if num_read<0:
        num_read+=100
    return num_read


if __name__ == '__main__':
    bboxs = np.array(
        [[311, 782, 327, 831], [338, 782, 359, 831], [416, 782, 430, 830], [439, 782, 462, 831], [524, 783, 537, 831],
         [549, 782, 570, 832], [634, 783, 648, 832], [659, 783, 680, 833]])
    category = np.array(['1', '4', '1', '5', '1', '6', '1', '7'])
    finger_bbox = np.array([469, 592, 543, 672])
    num_read_main(bboxs,category,finger_bbox)
