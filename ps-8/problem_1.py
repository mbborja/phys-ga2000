# import numpy as np
# import matplotlib.pyplot as plt

# def load_waveform(file_path):
#     # Load waveform data from the specified file
#     waveform_data = np.loadtxt(file_path)
#     return waveform_data

# def plot_waveform(waveform_data):
#     # Plot the waveform
#     plt.figure(figsize=(10, 4))
#     plt.plot(waveform_data)
#     plt.title('Waveform')
#     plt.xlabel('Sample')
#     plt.ylabel('Amplitude')
#     plt.show()

# def calculate_dft(waveform_data):
#     # Calculate the Discrete Fourier Transform (DFT)
#     dft_result = np.fft.fft(waveform_data)
#     return dft_result

# def plot_magnitudes(dft_result, num_coefficients=10000):
#     # Plot the magnitudes of the first num_coefficients DFT coefficients
#     magnitudes = np.abs(dft_result[:num_coefficients])
#     plt.figure(figsize=(10, 4))
#     plt.plot(magnitudes)
#     plt.title('Magnitudes of DFT Coefficients')
#     plt.xlabel('Coefficient Index')
#     plt.ylabel('Magnitude')
#     plt.show()

# if __name__ == "__main__":
#     # Specify the file path (either piano.txt or trumpet.txt)
#     file_path = "piano.txt"  # Change this to "trumpet.txt" if needed

#     # Load the waveform data
#     waveform_data = load_waveform(file_path)

#     # Plot the waveform
#     plot_waveform(waveform_data)

#     # Calculate the Discrete Fourier Transform (DFT)
#     dft_result = calculate_dft(waveform_data)

#     # Plot the magnitudes of the first 10,000 DFT coefficients
#     plot_magnitudes(dft_result, num_coefficients=10000)

import numpy as np
import matplotlib.pyplot as plt

def load_waveform(file_path):
    # Load waveform data from the specified file
    waveform_data = np.loadtxt(file_path)
    return waveform_data

def plot_waveform(waveform_data, instrument):
    # Plot the waveform
    time_axis = np.arange(len(waveform_data)) / 44100.0  # Time axis in seconds
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, waveform_data)
    plt.title('Waveform of ' + instrument)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(instrument+"_wf.png")
    plt.show()

def calculate_dft(waveform_data):
    # Calculate the Discrete Fourier Transform (DFT)
    dft_result = np.fft.fft(waveform_data)
    return dft_result

def plot_magnitudes(dft_result, instrument, num_coefficients=10000):
    # Plot the magnitudes of the first num_coefficients DFT coefficients
    magnitudes = np.abs(dft_result[:num_coefficients])
    frequency_axis = np.fft.fftfreq(len(dft_result), d=1/44100.0)  # Frequency axis in Hz
    plt.figure(figsize=(10, 4))
    plt.plot(frequency_axis[:num_coefficients], magnitudes)
    plt.title('Magnitudes of DFT Coefficients of ' + instrument)
    plt.xlabel('Frequency (Hz)')
    # plt.xlim(500,600)
    plt.axvline(x=522, color='red', linestyle='--', label='Freq = 522 Hz')
    plt.legend()
    plt.ylabel('Magnitude')
    plt.savefig(instrument+"_hz.png")
    plt.show()

if __name__ == "__main__":
    # Specify the file path (either piano.txt or trumpet.txt)
    instrument = "trumpet"
    file_path = instrument +".txt"  # Change this to "trumpet.txt" if needed

    # Load the waveform data
    waveform_data = load_waveform(file_path)

    # Plot the waveform
    plot_waveform(waveform_data, instrument)

    # Calculate the Discrete Fourier Transform (DFT)
    dft_result = calculate_dft(waveform_data)

    # Plot the magnitudes of the first 10,000 DFT coefficients
    plot_magnitudes(dft_result, instrument, num_coefficients=10000)
    