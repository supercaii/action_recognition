import logging, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from PIL import Image
from . import utils as U
import os


class Visualizer():
    def __init__(self, args):
        self.args = args
        U.set_logging(args)
        logging.info('')
        logging.info('Starting visualizing ...')

        self.action_names = {}
        # self.action_names['H4AM'] = [
        #     'Pick up/Place Carrier 1', 'Pick up/Place Gear Bearings 2', 'Pick up/Place Planet Gears 3', 'Pick up/Place Carrier Shaft 4',
        #     'Pick up/Place Sun Shaft 5', 'Pick up/Place Sun Gear 6', 'Pick up/Place Sun Gear Bearing 7', 'Pick up/Place Ring Bear 8',
        #     'Pick up Block 2 and place it on Block 1 9', 'Pick up/Place Cover 10', 'Pick up/Place Screws 11',
        #     'Pick up/Place Allen Key, Turn Screws, Return Allen Key and EGT 12'
        # ]
        self.action_names['H4AM'] = [
            '拾起/放下载体 1', '拾起/放下齿轮轴承 2', '拾起/放下行星齿轮 3', '拾起/放下承载轴 4',
            '拾起/放下太阳轴 5', '拾起/放下太阳齿轮 6', '拾起/放下太阳齿轮轴承 7', '拾起/放下环齿轮 8',
            '拿起组件2（动作5-8）并将其放置在组件1（动作1-4）上 9', '拾起/放下盖子 10', '拾起/放下螺丝 11',
            '拾起/放下内六角扳手、旋转螺丝、归还内六角扳手和 EGT 12'
        ]

        self.font_sizes = {
            'H4AM': 10
        }

    def start(self):
        self.read_data()
        logging.info('Please select visualization function from follows: ')
        logging.info('1) wrong sample (ws), 2) important joints (ij), 3) H4AM skeleton (ns),')
        logging.info('4) confusion matrix (cm), 5) action accuracy (ac)')
        while True:
            logging.info('Please input the number (or name) of function, q for quit: ')
            cmd = input(U.get_current_timestamp())
            if cmd in ['q', 'quit', 'exit', 'stop']:
                break
            elif cmd == '1' or cmd == 'ws' or cmd == 'wrong sample':
                self.show_wrong_sample()
            elif cmd == '2' or cmd == 'ij' or cmd == 'important joints':
                self.show_important_joints()
            elif cmd == '3' or cmd == 'ns' or cmd == 'NTU skeleton':
                self.show_NTU_skeleton()
            elif cmd == '4' or cmd == 'cm' or cmd == 'confusion matrix':
                self.show_confusion_matrix()
            elif cmd == '5' or cmd == 'ac' or cmd == 'action accuracy':
                self.show_action_accuracy()
            else:
                logging.info('Can not find this function!')
                logging.info('')

    def read_data(self):
        logging.info('Reading data ...')
        logging.info('')
        data_file = './visualization/extraction_{}.npz'.format(self.args.config)
        try:
            data = np.load(data_file)
        except:
            data = None
            logging.info('')
            logging.error('Error: Wrong in loading this extraction file: {}!'.format(data_file))
            logging.info('Please extract the data first!')
            raise ValueError()
        logging.info('*********************Video Name************************')
        logging.info(data['name'][self.args.visualization_sample])
        logging.info('')

        feature = data['feature'][self.args.visualization_sample]
        self.location = data['location']
        if len(self.location) > 0:
            self.location = self.location[self.args.visualization_sample]
        self.data = data['data'][self.args.visualization_sample]
        self.label = data['label']
        weight = data['weight']
        out = data['out']
        cm = data['cm']
        # 归一化混淆矩阵
        self.cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        dataset = self.args.dataset
        self.names = self.action_names[dataset]
        self.font_size = self.font_sizes[dataset]
        # 取每行最大值的索引
        self.pred = np.argmax(out, 1)
        self.pred_class = self.pred[self.args.visualization_sample] + 1
        self.actural_class = self.label[self.args.visualization_sample] + 1
        if self.args.visualization_class == 0:
            self.args.visualization_class = self.actural_class
        self.probablity = out[self.args.visualization_sample, self.args.visualization_class - 1]
        # CAM：特征可视化方法
        self.result = np.einsum('kc,ctvm->ktvm', weight, feature)
        # result(t,v,m)
        self.result = self.result[self.args.visualization_class - 1]

    def show_action_accuracy(self):
        cm = self.cm.round(4)

        logging.info('Accuracy of each class:')
        # 获取混淆矩阵对角线元素
        accuracy = cm.diagonal()
        for i in range(len(accuracy)):
            logging.info('{}: {}'.format(self.names[i], accuracy[i]))
        logging.info('')

        plt.figure()
        plt.bar(self.names, accuracy, align='center')
        plt.xticks(fontsize=10, rotation=90)
        plt.yticks(fontsize=10)
        plt.show()

    def show_confusion_matrix(self):
        cm = self.cm
        show_name_x = range(1, len(self.names) + 1)
        show_name_y = self.names

        plt.rcParams['font.sans-serif'] = ['SimSun']
        font_size = self.font_size
        sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='.3g', annot_kws={'fontsize': font_size - 3}, cbar=False,
                    square=True, linewidths=0.1, linecolor='black', xticklabels=show_name_x, yticklabels=show_name_x)
        plt.xticks(fontsize=font_size, rotation=0)
        plt.yticks(fontsize=font_size)
        plt.xlabel('预测类别', fontsize=font_size)
        plt.ylabel('真实类别', fontsize=font_size)
        plt.savefig('confusion_matrix.svg', bbox_inches='tight')
        plt.show()

    # def show_NTU_skeleton1(self):
    #     if len(self.location) == 0:
    #         logging.info('This function is only for NTU dataset!')
    #         logging.info('')
    #         return
    #
    #     C, T, V, M = self.location.shape
    #     connecting_joint = np.array(
    #         [1, 2, 3, 26, 2, 4, 5, 6, 7, 8, 7, 2, 11, 12, 13, 14, 15, 14, 0, 18, 19, 20, 0, 22, 23, 24, 27, 28, 30, 27,
    #          28, 29])
    #     # 归一化
    #     result = np.maximum(self.result, 0)
    #     result = result / np.max(result)
    #
    #     if len(self.args.visualization_frames) > 0:
    #         pause, frames = 10, self.args.visualization_frames
    #     else:
    #         pause, frames = 0.1, range(self.location.shape[1])
    #
    #     plt.figure()
    #     plt.ion()
    #     for t in frames:
    #         if np.sum(self.location[:, t, :, :]) == 0:
    #             break
    #
    #         plt.cla()
    #         plt.xlim(-50, 2000)
    #         plt.ylim(-50, 1100)
    #         plt.axis('off')
    #         plt.title('sample:{}, class:{}, frame:{}\n probablity:{:2.2f}%, pred_class:{}, actural_class:{}'.format(
    #             self.args.visualization_sample, self.names[self.args.visualization_class - 1],
    #             t, self.probablity * 100, self.pred_class, self.actural_class,
    #         ), fontproperties="SimHei")
    #
    #         for m in range(M):
    #             x = self.location[0, t, :, m]
    #             y = 1080 - self.location[1, t, :, m]
    #
    #             c = []
    #             for v in range(V):
    #                 r = result[t // 4, v, m]
    #                 g = 0
    #                 b = 1 - r
    #                 c.append([r, g, b])
    #                 k = connecting_joint[v]
    #                 # 绘制人体骨骼
    #                 plt.plot([x[v], x[k]], [y[v], y[k]], '-o', c=np.array([0.1, 0.1, 0.1]), linewidth=0.5, markersize=0)
    #             # 散点图
    #             plt.scatter(x, y, marker='o', c=c, s=20)
    #         plt.pause(pause)
    #     plt.ioff()
    #     plt.show()

    def show_NTU_skeleton(self):
        C, T, V, M = self.location.shape
        connecting_joint = np.array(
            [1, 2, 3, 26, 2, 4, 5, 6, 7, 8, 7, 2, 11, 12, 13, 14, 15, 14, 0, 18, 19, 20, 0, 22, 23, 24, 27, 28, 30, 27,
             28, 29])
        # 归一化
        result = np.maximum(self.result, 0)
        result = result / np.max(result)

        if len(self.args.visualization_frames) > 0:
            pause, frames = 10, self.args.visualization_frames
        else:
            pause, frames = 0.1, range(self.location.shape[1])

        root_dir = "D:/CaiZeXiong/action_recognition/P07A05"
        colAStruct = os.listdir(root_dir)
        for t in frames:
            if np.sum(self.location[:, t, :, :]) == 0:
                break
            img_dir = os.path.join(root_dir, colAStruct[t])
            img = Image.open(img_dir)
            plt.imshow(img)
            plt.axis('off')
            plt.title('sample:{}, class:{}, frame:{}\n probablity:{:2.2f}%, pred_class:{}, actural_class:{}'.format(
                self.args.visualization_sample, self.names[self.args.visualization_class - 1],
                t, self.probablity * 100, self.pred_class, self.actural_class
            ), fontproperties="SimHei")

            x = self.location[0, t, :, 0]
            y = self.location[1, t, :, 0]

            c = []
            for v in range(V):
                # r = result[t // 4, v, 0]
                r = result[t, v, 0]
                g = 0
                b = 1 - r
                c.append([r, g, b])
                k = connecting_joint[v]
                plt.plot([x[v], x[k]], [y[v], y[k]], '-', c=np.array([0.1, 0.1, 0.1]), linewidth=0.5, markersize=0)
            plt.scatter(x, y, marker='o', c=c, s=40)
            plt.draw()
            plt.savefig("./result/"+str(t) + ".png", bbox_inches='tight', pad_inches=0.0)
            plt.pause(pause)
            plt.clf()

    # plt.ioff()
    # plt.show()

    def show_wrong_sample(self):
        wrong_sample = []
        for i in range(len(self.pred)):
            if not self.pred[i] == self.label[i]:
                wrong_sample.append(i)
        logging.info('*********************Wrong Sample**********************')
        logging.info(wrong_sample)
        logging.info('')

    def show_important_joints(self):
        first_sum = np.sum(self.result[:, :, 0], axis=0)
        first_index = np.argsort(-first_sum) + 1
        logging.info('*********************First Person**********************')
        logging.info('Weights of all joints:')
        logging.info(first_sum)
        logging.info('')
        logging.info('Most important joints:')
        logging.info(first_index)
        logging.info('')

        if self.result.shape[-1] > 1:
            second_sum = np.sum(self.result[:, :, 1], axis=0)
            second_index = np.argsort(-second_sum) + 1
            logging.info('*********************Second Person*********************')
            logging.info('Weights of all joints:')
            logging.info(second_sum)
            logging.info('')
            logging.info('Most important joints:')
            logging.info(second_index)
            logging.info('')
