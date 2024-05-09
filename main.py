import librosa
import numpy as np
import matplotlib.pyplot as plt

# Function to run the process and return the random numbers
def process_audio():
    # Load audio file
    audio, sr = librosa.load('sample.wav')
    
    # Extract features
    features = librosa.feature.mfcc(y=audio, sr=sr)
    
    # Generate random numbers
    random_numbers = np.random.rand(features.shape[1])

    # choose one random number from the list
    random_number = random_numbers[np.random.randint(0, len(random_numbers))]
    print(random_number)

    # print(*features.shape)

    # save random numbers to a file
    random_numbers = np.round(random_numbers * 100) / 100
    
    return random_numbers

random_numbers_list = [process_audio() for _ in range(1)]

# Plot the results
plt.figure(figsize=(10, 6))
for i, random_numbers in enumerate(random_numbers_list, start=1):
    plt.plot(random_numbers, label=f'Run {i}')

plt.title('Random Numbers from Audio Processing')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Save the plot as an image
plt.savefig('random_numbers_plot.png', dpi=300)

plt.show()
