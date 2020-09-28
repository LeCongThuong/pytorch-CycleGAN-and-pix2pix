import re
import matplotlib.pyplot as plt


def load_file_to_list(file_path):
    return [line.rstrip('\n') for line in open(file_path)]


def plot_loss(loss_log_file: str):
    content = load_file_to_list(loss_log_file)
    g_gan_loss = []
    g_l1_loss = []
    d_real_loss = []
    d_fake_loss = []
    g_gan_loss_pattern = "G_GAN:\s+(.{4,5})"
    g_l1_loss_pattern = "G_L1:\s+(\d{1,2}.\d{3})"
    d_real_loss_pattern = "D_real:\s+(\d{1,2}.\d{3})"
    d_fake_loss_pattern = "D_fake:\s+(\d{1,2}.\d{3})"
    iter_pattern = "iters:\s+(\d{2,5})"

    for line in content:
        if 'epoch' in line:
            try:
                iter_num = int(re.search(iter_pattern, line).group(1))
            except Exception as e:
                print(line)
                exit(0)

            if iter_num % 1000 == 0:
                g_gan_loss.append(re.search(g_gan_loss_pattern, line).group(1))
                g_l1_loss.append(re.search(g_l1_loss_pattern, line).group(1))
                d_real_loss.append(re.search(d_real_loss_pattern, line).group(1))
                d_fake_loss.append(re.search(d_fake_loss_pattern, line).group(1))
    print(len(g_gan_loss))
    fig, ax = plt.subplots(figsize=(12, 9))
    x_axis = range(len(g_gan_loss))
    # ax.plot(x_axis, g_gan_loss, label="GAN Loss of Generator")
    # ax.plot(x_axis, g_l1_loss, label="GAN L1 Loss of Generator")
    ax.plot(x_axis, d_real_loss, label="GAM Real Loss of Discriminator")
    # ax.plot(x_axis, d_fake_loss, label="GAN Fake Loss of Discriminator")
    plt.show()


if __name__ == '__main__':
    plot_loss("/home/love_you/Downloads/loss_log.txt")
