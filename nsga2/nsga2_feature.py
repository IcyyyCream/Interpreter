import copy
import math

import numpy as np
import random

from nsga2.nsga2 import Population
from model.getfeatures import get_features, get_prediction, mask_features


class PopulationFeature(Population):
    """
    Explanation population of NSGA2 for Feature
    """

    def __init__(self, image, population_size=50, label_index=None):
        """
        Initialize population.

        :param image: (3D PIL, ndarray, tensor) Input image
        :param population_size: (int) Population size, default is 100
        :param label_index: (int) The label index to be explained, default is top1
        """

        # 保留原始特征图
        self.original_features, self.gradients = get_features(image)
        # 消融实验
        # self.gradients = np.ones(self.gradients.shape)
        # 副本
        features = self.original_features
        # 个体大小
        individual_size = features.size()[1]

        super().__init__(image, population_size, individual_size, label_index)

    def generate(self):
        """
        Randomly generate individual vectors.

        :return: (2D array) Initial random population (Each row represents an individual)
        """

        # 随机生成的0.0-1.0均匀分布的向量与梯度向量相乘
        population = self.gradients * np.random.random_sample((self.population_size, self.individual_size))

        return population

    def functions(self, population):
        """
        Functions to optimize, i.e.
        1. the absolute value of prediction difference;
        2. the norm of an individual vector.

        :param population: (2D array) Population to be calculated (Each row represents an individual)
        :return: (1D list: double) Module prediction value, (1D list: int) Individual vector norm
        """

        prediction = []
        norm = []
        # 遍历种群
        for ind in population:
            # 生成遮罩特征图
            new_features = mask_features(self.original_features, ind)
            # 送去网络预测
            pre = get_prediction(new_features)[0]
            # 优化函数1的值
            # 普通形式
            prediction.append(abs(pre[self.label_index] - self.prediction_value))

            # 消融实验
            # prediction.append(1)

            # 优化函数2的值
            temp = 0
            for val in ind:
                if val != 0:
                    temp += 1
            # 普通形式
            norm.append(temp)

            # 消融实验
            # norm.append(1)

        return prediction, norm

    def select(self):
        """
        Select population individuals for crossover, mutation operations.

        :return: (1D list) Selected index
        """

        aggregation = []
        for ind in self.population:
            # 初始化聚集度为0
            agg = 0
            temp = 0
            for i in range(self.individual_size):
                if ind[i] != 0:
                    temp += 1
                    if i == self.individual_size - 1:
                        agg += temp * temp

                else:
                    agg += temp * temp
                    temp = 0
            aggregation.append(agg)

        select_pool = list(range(self.population_size))
        selected_num = 0
        selected_ind = []
        # 共选择种群大小一半的个体
        while selected_num < int(self.population_size / 2):
            # 锦标赛选择, 每次选出5个中的最小(有放回)
            choice = random.sample(select_pool, 5)
            winner = min(choice, key=lambda _: aggregation[_])
            # 消融实验
            # winner = choice[0]
            selected_ind.append(winner)
            select_pool.remove(winner)
            selected_num += 1

        return selected_ind

    @staticmethod
    def crossover(individual1, individual2):
        """
        Analog binary crossover.

        :param individual1: (1D array) crossover father
        :param individual2: (1D array) crossover mather
        :return: (2 * 1D array) Two individuals after crossover
        """

        # 个体大小
        ind_len = len(individual1)
        # 指数分布常数
        eta = 10
        # 在[0, 1)之间按均匀分布取随机数
        mu = np.random.uniform(0.0, 1.0, 1)
        # 计算扩展因子beta
        if mu < 0.5:
            beta = (2 * mu) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 - 2 * mu)) ** (1 / (eta + 1))

        offspring1 = 0.5 * ((1 + beta) * individual1 + (1 - beta) * individual2)
        offspring2 = 0.5 * ((1 - beta) * individual1 + (1 + beta) * individual2)
        # 控制个体值符合0.5~2.0区间
        for i in range(ind_len):
            if offspring1[i] < 0.5:
                offspring1[i] = 0
            # if offspring1[i] > 2:
            #     offspring1[i] = 2
            if offspring2[i] < 0.5:
                offspring2[i] = 0
            # if offspring2[i] > 2:
            #     offspring2[i] = 2

        return offspring1, offspring2

    def mutation(self, individual1, individual2):
        """
        Perform mutation operation on an individual.

        :param individual1: (1D array) Individuals to be mutated
        :param individual2: (1D array) Individuals to be mutated
        :return: (2 * 1D array) Two individuals after mutation
        """

        ind_len = len(individual1)
        # 拷贝一份个体
        offspring1 = copy.deepcopy(individual1)
        offspring2 = copy.deepcopy(individual2)
        # 随机选取1/10个位置变异
        count = int(ind_len / 10)
        mutate_loc1 = random.sample(range(ind_len), count)
        mutate_loc2 = random.sample(range(ind_len), count)
        # 变异
        for loc1, loc2 in zip(mutate_loc1, mutate_loc2):
            offspring1[loc1] = self.gradients[loc1]
            offspring2[loc2] = self.gradients[loc2]
        # # 控制个体值符合0.5~2.0区间
        for i in range(ind_len):
            if offspring1[i] < 0.5:
                offspring1[i] = 0
            # if offspring1[i] > 2:
            #     offspring1[i] = 2
            if offspring2[i] < 0.5:
                offspring2[i] = 0
            # if offspring2[i] > 2:
            #     offspring2[i] = 2

        return offspring1, offspring2

    def scm(self):
        """
        (s-selection, c-crossover, m-mutation)
        Randomly select individuals in the population for crossover and mutation to generate offspring.

        :return: (2D array) Offspring population, each row represents an individual
        """

        offsprings = []
        pool = self.select()
        while len(pool) >= 2:
            index1 = pool.pop()
            index2 = pool.pop()
            # 变异操作
            if np.random.choice([0, 1], p=[0.80, 0.20]) == 1:
                off1, off2 = self.mutation(self.population[index1], self.population[index2])
                offsprings.append(off1)
                offsprings.append(off2)
            # 交叉操作
            else:
                off1, off2 = self.crossover(self.population[index1], self.population[index2])
                offsprings.append(off1)
                offsprings.append(off2)

        return np.array(offsprings)


if __name__ == '__main__':
    # img = imread('super_pixel/image/wan.jpg')
    # seg = generate_segment(img, 'SLIC')
    #
    # pop1 = Population(img, seg)
    # pre, n = pop1.functions()
    #
    # pa = pop1.non_dominated_sort(pre, n)
    # print(pa)
    # print(pop1.crowding_sort(pre, n, pa[0]))
    print('yes')
