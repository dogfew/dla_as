import io
import matplotlib.pyplot as plt


def plot_spectrogram_to_buf(spectrogram_tensor, name=None):
    plt.figure(figsize=(20, 4))
    plt.imshow(spectrogram_tensor)
    plt.title(name)
    plt.tight_layout()
    # plt.subplots_adjust(top=1, bottom=0.)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    # plt.show()
    plt.close()
    return buf
