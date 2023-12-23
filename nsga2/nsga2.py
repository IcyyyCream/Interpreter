import abc
import numpy as np
import itertools

from model.detect import detect_via_image


class Population:
    """
    Pure virtual base class of NSGA2 population, You need to inherit this class and rewrite its methods in subclasses.
    """

    def __init__(self, image, population_size=100, individual_size=100, label_index=None):
        """
        Initialize population.

        :param image: (3D PIL, ndarray, tensor) Input image
        :param population_size: (int) Population size, default is 100
        :param individual_size: (int) Individual size, default is 100
        :param label_index: (int) The label index to be explained, if None, select top1
        """

        # 保留原始图像
        self.original_image = image
        # 个体大小
        self.individual_size = individual_size
        # 种群大小
        self.population_size = population_size
        # 随机创造个体向量, 生成种群
        # self.population = np.random.choice([0, 1], size=(self.population_size, self.individual_size), p=[0.7, 0.3])
        self.population = self.generate()
        # 进行一次预测, 初始化待解释的标签概率值及其索引
        # 模型预测输出是(1×1000)的2D矩阵, prediction[0]才是预测的1000维概率向量
        prediction = detect_via_image(self.original_image)[0]
        if label_index is None:
            self.label_index = np.argmax(prediction)
            self.prediction_value = prediction[self.label_index]
        else:
            self.label_index = label_index
            self.prediction_value = prediction[label_index]
        # 获得种群中每个个体对应的两个目标函数的值
        self.f1_value, self.f2_value = self.functions(self.population)
        self.pareto_list = self.non_dominated_sort()

    @abc.abstractmethod
    def generate(self):
        """
        The population random generation method, needs to be rewritten.
        """
        pass

    @abc.abstractmethod
    def functions(self, population):
        """
        Functions to optimize, needs to be rewritten.
        """
        pass

    def non_dominated_sort(self):
        """
        Perform a fast non dominated sorting on the population.

        :return: (2D list) Index list with Pareto rank
        """

        # 个体被支配个数n_p
        np_list = [0] * self.population_size
        # 个体支配的解的集合s_p
        sp_list = []
        # 帕累托等级排序f1、f2···
        pareto_list = []
        # 排序终止判断
        rest = self.population_size

        # 计算种群中每个个体的两个参数n_p和s_p
        for i in range(self.population_size):
            sp = []
            for j in range(self.population_size):
                if j == i:
                    continue
                if self.f1_value[i] <= self.f1_value[j] and self.f2_value[i] <= self.f2_value[j] \
                        and (self.f1_value[i] < self.f1_value[j] or self.f2_value[i] < self.f2_value[j]):
                    np_list[j] += 1
                    sp.append(j)
            sp_list.append(sp)

        # 将种群中参数n_p=0的个体放入集合f1中
        temp = []
        for i, n_p in enumerate(np_list):
            if n_p == 0:
                temp.append(i)
                rest -= 1
        pareto_list.append(temp)

        i = 0
        # 判断是否划分完毕
        while rest:
            temp = []
            # 消除帕累托等级1对其余个体的支配, 并生成f2, 以此类推
            for k in pareto_list[i]:
                for t in sp_list[k]:
                    np_list[t] -= 1
                    if np_list[t] == 0:
                        temp.append(t)
                        rest -= 1
            pareto_list.append(temp)
            i += 1

        return pareto_list

    def crowding_sort(self, pareto_set):
        """
        Rank individuals in a specific Pareto set in descending order of crowding degree.

        :param pareto_set: (1D list) Index of a specific Pareto set
        :return: (1D list) The same Pareto set, except the crowding degree is in descending order
        """

        # 初始化该帕累托等级下的个体拥挤度为0
        crowd = dict(zip(pareto_set, [0] * len(pareto_set)))
        # 取得该等级下个体的目标函数值
        f1_p = [self.f1_value[_] for _ in pareto_set]
        f2_p = [self.f2_value[_] for _ in pareto_set]
        # 生成与索引对应的值的字典
        f1_dict = dict(zip(pareto_set, f1_p))
        f2_dict = dict(zip(pareto_set, f2_p))
        # 对字典的值进行排序, 返回索引-函数值的元组
        index_value1 = sorted(f1_dict.items(), key=lambda _: _[1])
        frac1 = index_value1[-1][1] - index_value1[0][1]
        index_value2 = sorted(f2_dict.items(), key=lambda _: _[1])
        frac2 = index_value2[-1][1] - index_value2[0][1]
        # 对目标函数f1计算拥挤度
        for i in range(len(pareto_set)):
            if i == 0 or i == len(pareto_set)-1:
                crowd[index_value1[i][0]] = 10000
            else:
                crowd[index_value1[i][0]] += (index_value1[i+1][1] - index_value1[i-1][1]) / (frac1 + 0.01)

        # 对目标函数f2计算拥挤度
        for i in range(len(pareto_set)):
            if i == 0 or i == len(pareto_set)-1:
                continue
            else:
                crowd[index_value2[i][0]] += (index_value2[i+1][1] - index_value2[i-1][1]) / (frac2 + 0.01)
        return sorted(crowd.keys(), key=crowd.__getitem__, reverse=True)

    @abc.abstractmethod
    def select(self):
        """
        Select method, needs to be rewritten.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def crossover(*args):
        """
        Crossover method, needs to be rewritten.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def mutation(*args):
        """
        Mutation method, needs to be rewritten.
        """
        pass

    @abc.abstractmethod
    def scm(self):
        """
        (s-selection, c-crossover, m-mutation)
        Generate offsprings, needs to be rewritten.
        """
        pass

    def regeneration(self, offsprings):
        """
        Merge parents and offsprings to generate a new parents and renew the population.

        :param offsprings: (2D array) Offspring generation after selection, crossover and mutation
        :return: None
        """

        # 记录原始种群大小
        origin_size = self.population_size
        # 扩充种群, 重新计算种群大小、目标函数值、帕累托等级
        if len(offsprings) == 0:
            expanded_population = self.population
        else:
            expanded_population = np.vstack((self.population, offsprings))
        self.population = expanded_population
        self.population_size = len(expanded_population)
        new_f1_value, new_f2_value = self.functions(offsprings)
        self.f1_value += new_f1_value
        self.f2_value += new_f2_value
        self.pareto_list = self.non_dominated_sort()

        new_population_index = []
        new_population = []
        # 新种群中的剩余位置
        rest = origin_size
        # 按照帕累托等级和拥挤度选择个体索引
        for pareto in self.pareto_list:
            if rest - len(pareto) >= 0:
                new_population_index.append(pareto)
                rest -= len(pareto)
            else:
                pareto = self.crowding_sort(pareto)
                new_population_index.append(pareto[0:rest])
                break
        # 2D列表展成1D
        new_population_index = list(itertools.chain(*new_population_index))
        # 选择个体组成新父代
        for index in new_population_index:
            new_population.append(self.population[index])
        # 更新种群
        self.population = np.array(new_population)
        self.population_size = origin_size
        self.f1_value = [self.f1_value[i] for i in new_population_index]
        self.f2_value = [self.f2_value[i] for i in new_population_index]
        self.pareto_list = self.non_dominated_sort()


if __name__ == '__main__':
    print('yes')
