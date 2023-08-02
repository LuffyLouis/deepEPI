import argparse

# 创建ArgumentParser对象
parser = argparse.ArgumentParser()

# 添加参数和参数组
parser.add_argument('--mode', choices=['train', 'test'], help='选择模式：train或test')
group = parser.add_argument_group('训练参数')
group.add_argument('--lr', type=float, help='学习率')
group.add_argument('--epochs', type=int, help='训练的轮数')
group = parser.add_argument_group('测试参数')
group.add_argument('--model_path', help='模型文件路径')

# 解析命令行参数
args = parser.parse_args()

# 根据参数值进行逻辑控制
if args.mode == 'train':
    if not (args.lr and args.epochs):
        parser.error('训练模式需要指定学习率和轮数')
    else:
        # 执行训练逻辑
        print('开始训练，学习率：{}，轮数：{}'.format(args.lr, args.epochs))
elif args.mode == 'test':
    if not args.model_path:
        parser.error('测试模式需要指定模型文件路径')
    else:
        # 执行测试逻辑
        print('开始测试，模型文件路径：{}'.format(args.model_path))
