# import ikpy.chain
# import numpy as np
# import ikpy.utils.plot as plot_utils
#
# my_chain = ikpy.chain.Chain.from_urdf_file("poppy_ergo.URDF")
#
# target_position = [0.5, -0.2, 0.5]
# print("The angles of each joints are: ", my_chain.inverse_kinematics(target_position))
#
# real_frame = my_chain.forward_kinematics(my_chain.inverse_kinematics(target_position))
# print("Computed position vector : %s, original position vector : %s" % (real_frame[:3, 3], target_position))
#
# import matplotlib.pyplot as plt
# fig, ax = plot_utils.init_3d_figure()
# my_chain.plot(my_chain.inverse_kinematics(target_position), ax, target=target_position)
# plt.xlim(-0.1, 0.1)
# plt.ylim(-0.1, 0.1)
# plt.show()

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import matplotlib.pyplot as plt
import ikpy.utils.plot as plot_utils

# Определение звеньев
link1 = URDFLink(
    name="link1",
    bounds=(-np.pi, np.pi),
    origin_translation=np.array([0.0, -0.5, 3]),
    origin_orientation=np.array([0.0, 0.0, 0.0]),
    rotation=np.array([0.0, 1.0, 0.0]),
    joint_type="revolute"
)

# link2 = URDFLink(
#     name="link2",
#     bounds=(-np.pi, np.pi),
#     origin_translation=np.array([0.0, 0.0, 2]),
#     origin_orientation=np.array([0.0, 0.0, 0.0]),
#     rotation=np.array([0.0, 1.0, 0.0]),
#     joint_type="revolute"
# )
#
# link3 = URDFLink(
#     name="link3",
#     bounds=(-np.pi, np.pi),
#     origin_translation=np.array([0.0, 0.0, 0.5]),
#     origin_orientation=np.array([0.0, 0.0, 0.0]),
#     rotation=np.array([0.0, 0.0, 1.0]),
#     joint_type="revolute"
# )

# Создание кинематической цепи
my_chain = Chain(links=[
    OriginLink(),
    link1
    # link2
    # link3
])

# Решение обратной задачи кинематики
target_position = [0.5, 0.5, 0.5]
target_orientation = [1, 0.5, 0.5, 0.5]  # Вектор ориентации (кватернион)

#____________________________________________________________ дядя сережа вставил_______________________________-
# from ikpy.chain import Chain
# from ikpy.link import OriginLink, URDFLink
# import numpy as np
# import matplotlib.pyplot as plt
# import ikpy.utils.plot as plot_utils

# # Определение звеньев
# link1 = URDFLink(
#     name="link1",
#     bounds=(-np.pi, np.pi),
#     origin_translation=np.array([0.0, 0.0, 0.5]),
#     origin_orientation=np.array([0.0, 0.0, 0.0]),
#     rotation=np.array([0.0, 0.0, 1.0]),
#     joint_type="revolute"
# )

# link2 = URDFLink(
#     name="link2",
#     bounds=(-np.pi, np.pi),
#     origin_translation=np.array([0.0, 0.5, 0.5]),
#     origin_orientation=np.array([-0.5, 0.0, 0.0]),
#     rotation=np.array([0.0, 0.0, 1.0]),
#     joint_type="revolute"
# )

# link3 = URDFLink(
#     name="link3",
#     bounds=(-np.pi, np.pi),
#     origin_translation=np.array([0.0, 0.5, 0.0]),
#     origin_orientation=np.array([0.0, 0.0, 0.0]),
#     rotation=np.array([0.0, 0.0, 1.0]),
#     joint_type="revolute"
# )

# # Создание кинематической цепи
# my_chain = Chain(links=[
#     OriginLink(),
#     link1,
#     link2,
#     link3
# ])

# # Решение обратной задачи кинематики

# # Решение обратной задачи кинематики
# target_position = [0.5, 0.5, 0.5]
# target_orientation = [1, 0.5, 0.5, 0.5]

# joint_angles = my_chain.inverse_kinematics(target_position, target_orientation)
# print("Углы соединений:", joint_angles)

# # Визуализация цепи
# fig, ax = plot_utils.init_3d_figure()
# my_chain.plot(my_chain.inverse_kinematics(target_position), ax, target=target_position)
# plt.xlim(-2.1, 2.1)
# plt.ylim(-2.1, 2.1)

# # Создание кинематической цепи
# my_chain = Chain(links=[
#     OriginLink(),
#     link1,
#     link2,
#     link3
# ])

# # Генерация сетки углов соединений
# num_samples = 30
# theta1 = np.linspace(-np.pi, np.pi, num_samples)
# theta2 = np.linspace(-np.pi, np.pi, num_samples)
# theta3 = np.linspace(-np.pi, np.pi, num_samples)

# # Инициализация массива для хранения позиций конечного звена
# end_effector_positions = []

# # Вычисление позиций конечного звена для каждой комбинации углов
# for t1 in theta1:
#     for t2 in theta2:
#         for t3 in theta3:
#             joint_angles = [0, t1, t2, t3]
#             end_effector_position = my_chain.forward_kinematics(joint_angles)[:3, 3]
#             end_effector_positions.append(end_effector_position)

# # Преобразование списка в numpy массив
# end_effector_positions = np.array(end_effector_positions)

# # Визуализация области работы
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(end_effector_positions[:, 0], end_effector_positions[:, 1], end_effector_positions[:, 2], c='b', marker='.')

# plt.show()
