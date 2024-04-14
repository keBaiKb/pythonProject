
import pywt,cv2 as cv,numpy as n

class VIFusion:
    # 融合策略
    def strategy(self,arr_hvd1,arr_hvd2):
        # 小波系数融合
        k1 = 0.8
        k2 = 0.2
        arr_w1 = n.where( n.abs(arr_hvd1) > n.abs(arr_hvd2), k1, k2)
        arr_w2 = n.where( n.abs(arr_hvd1) < n.abs(arr_hvd2), k1, k2)
        return arr_w1,arr_w2

    # 融合
    def fusion(self, arr_visible, arr_infrared):
        # 进行小波变换的图像的长宽必须为偶数
        it_h1 = arr_visible.shape[0]
        it_w1 = arr_visible.shape[1]
        it_h2 = arr_infrared.shape[0]
        it_w2 = arr_infrared.shape[1]
        if it_h1 % 2 != 0:
            it_h1 = it_h1 + 1
        if it_w1 % 2 != 0:
            it_w1 = it_w1 + 1
        if it_h2 % 2 != 0:
            it_h2 = it_h2 + 1
        if it_w2 % 2 != 0:
            it_w2 = it_w2 + 1
        arr_visible = cv.resize(arr_visible, (it_w1, it_h1))
        arr_infrared = cv.resize(arr_infrared, (it_w2, it_h2))

        # 小波分解层数
        it_level = 5

        # 对两个图像的灰度图进行小波变换
        arr_Gray1,arr_Gray2 = cv.cvtColor(arr_visible, cv.COLOR_BGR2GRAY), cv.cvtColor(arr_infrared, cv.COLOR_BGR2GRAY)

        # 类型转换
        arr_Gray1 = arr_Gray1*1.0
        arr_Gray2 = arr_Gray2*1.0

        arr_visible = arr_visible*1.0
        arr_infrared = arr_infrared*1.0

        # 分别对 可见光的灰度图 和 红外光的灰度图 进行小波变换
        arr_decGray1 = pywt.wavedec2(arr_Gray1, 'sym4', level=it_level)
        arr_decGray2 = pywt.wavedec2(arr_Gray2, 'sym4', level=it_level)

        # 对可见光进行 RGB 进行小波变换
        ls_decRed1 = pywt.wavedec2(arr_visible[:, :, 0], 'sym4', level=it_level)
        ls_decGreen1 = pywt.wavedec2(arr_visible[:, :, 1], 'sym4', level=it_level)
        ls_decBlue1 = pywt.wavedec2(arr_visible[:, :, 2], 'sym4', level=it_level)

        # 重构列表
        ls_recRed = []
        ls_recGreen = []
        ls_recBlue = []

        for it_i,(arr_gray1,arr_gray2,arr_red1,arr_green1,arr_blue1) in enumerate(zip(arr_decGray1,arr_decGray2,ls_decRed1,ls_decGreen1,ls_decBlue1)):
            if it_i == 0:
                # 低频
                # 平均加权融合
                fl_w1 = 0.5
                fl_w2 = 0.5
                us_recRed = fl_w1 * arr_red1 + fl_w2 * arr_gray2
                us_recGreen = fl_w1 * arr_green1 + fl_w2 * arr_gray2
                us_recBlue = fl_w1 * arr_blue1 + fl_w2 * arr_gray2
            else:
                # 高频
                #基于绝对值最大选取融合系数
                us_recRed = []
                us_recGreen = []
                us_recBlue = []
                for arr_grayHVD1,arr_grayHVD2,arr_redHVD1,arr_greenHVD1,arr_blueHVD1, in zip(arr_gray1,arr_gray2,arr_red1,arr_green1,arr_blue1):
                    arr_w1,arr_w2 = self.strategy(arr_grayHVD1,arr_grayHVD2)
                    arr_recRed = arr_w1*arr_redHVD1 + arr_w2*arr_grayHVD2
                    arr_recGreen = arr_w1*arr_greenHVD1 + arr_w2*arr_grayHVD2
                    arr_recBlue = arr_w1*arr_blueHVD1 + arr_w2*arr_grayHVD2

                    us_recRed.append(arr_recRed)
                    us_recGreen.append(arr_recGreen)
                    us_recBlue.append(arr_recBlue)

            # 重构列表添加
            ls_recRed.append(us_recRed)
            ls_recGreen.append(us_recGreen)
            ls_recBlue.append(us_recBlue)

        # 重构
        arr_rec = n.zeros(arr_visible.shape)
        arr_rec[:, :, 0] = pywt.waverec2(ls_recRed, 'sym4')
        arr_rec[:, :, 1] = pywt.waverec2(ls_recGreen, 'sym4')
        arr_rec[:, :, 2] = pywt.waverec2(ls_recBlue, 'sym4')

        return arr_rec












