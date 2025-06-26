# Self-Driving Car Neural Network Simulation with DQN (No Libraries)

This project extends and builds upon the excellent self-driving car simulation by [gniziemazity](https://github.com/gniziemazity/Self-driving-car), adding complete Deep Q-Learning (DQN) capabilities and reinforcement learning environment from scratch.

## Project Extension Overview

While the original project provides the foundational car mechanics, road definition, and basic neural network implementation, this extension adds:

### New From-Scratch Implementations:
1. **Complete DQN Architecture**
   - Custom Deep Q-Network implementation
   - Experience replay buffer
   - Target network mechanism
   - TD-error computation
   - ε-greedy exploration strategy

2. **Reinforcement Learning Environment**
   - Custom `CarRLEnvironment` class
   - State and reward management
   - Episode handling
   - Traffic generation
   - Training loop implementation

3. **Advanced Training Mechanisms**
   - Batch processing
   - Gradient accumulation
   - Custom backpropagation
   - Parameter update strategies

## Technical Details

### State Space
- 5 sensor readings (normalized to [0,1])
- Car speed
- Car angle
- Distance to finish line
- Lane position

### Action Space
4 discrete actions:
1. Forward
2. Forward + Left
3. Forward + Right
4. No Movement

### Reward Structure
- Progress reward for moving forward
- Speed reward for maintaining velocity
- Crash penalty for collisions
- Finish line bonus
- Lane-keeping reward
- Time penalty for efficiency

## Getting Started

1. Clone the repository
2. Open `index.html` in a modern web browser
3. The simulation will start automatically

### Training Mode
```javascript
// Initialize training environment
const env = new CarRLEnvironment(-5000, "AI");
const policy = new DQN([5,16,4]);
```

### Manual Control Mode
```javascript
// Initialize manual control
const car = new Car(road.getLaneCenter(1), 100, 30, 50, "Keys");
```

## Configuration Options

### Sensor Configuration
```javascript
// in sensor.js
this.rayCount = 5;      // Number of sensor rays
this.rayLength = 300;   // Length of sensor rays
this.raySpread = Math.PI/4;  // Angle spread of rays
```

### Training Parameters
```javascript
let epsilon = 0.2;        // Exploration rate
let batch_size = 32;      // Training batch size
let num_episodes = 100;   // Number of training episodes
```

### Environment Settings
```javascript
const FINISH_LINE_Y = -5000;  // Distance to finish line
const MAX_STEPS = 10000;      // Steps per episode
```

## Neural Network Implementation Details

### Network Architecture
```javascript
class NeuralNetwork {
    constructor(neuronCounts) {
        this.levels = [];
        for (let i = 0; i < neuronCounts.length - 1; i++) {
            this.levels.push(new Level(neuronCounts[i], neuronCounts[i + 1]));
        }
    }
}
```

### DQN Core Components
```javascript
class DQN {
    constructor(layerSizes) {
        this.network = new NeuralNetwork(layerSizes);
        this.target = this.clone();
        this.replayBuffer = [];
    }
}
```

### RL Environment (New Addition)
```javascript
class CarRLEnvironment {
    constructor(finishLineY, controlType) {
        // Environment configuration
        this.FINISH_LINE_Y = finishLineY;
        this.ACTION_SPACE = 4;
        this.STATE_SPACE = 5;
        
        // Reward structure
        this.CRASH_PENALTY = -500;
        this.FINISH_REWARD = 10000;
        this.PROGRESS_REWARD_SCALE = 100.0;
    }
}
```

## Files Structure
```text
self-driving-car-NN/
├── index.html # Main HTML file
├── style.css # Styling
├── main-2.js # Training implementation
├── car.js # Car physics and logic
├── sensor.js # Sensor implementation
├── road.js # Road environment
├── controls.js # Input handling
├── network.js # Neural network base
├── dqn_network.js # DQN implementation
├── rl_environment.js # RL environment
├── visualizer.js # Network visualization
└── utils.js # Helper functions
```


All new components are implemented entirely in vanilla JavaScript with no external machine learning libraries.

## Original Project Components Used:
- Car physics and controls
- Road and sensor implementation
- Basic collision detection
- Traffic simulation base
- Visualization framework

## Key Differences from Original Project

1. **Learning Approach**
   - Original: Genetic algorithm/evolutionary approach
   - This Extension: Deep Q-Learning with experience replay

2. **Training Method**
   - Original: Population-based training
   - This Extension: Single agent with temporal difference learning

3. **State Representation**
   - Original: Direct sensor readings
   - This Extension: Enhanced state space with additional features

4. **Action Space**
   - Original: Binary outputs for controls
   - This Extension: Discrete action space with four possible actions

## Features

- Real-time visualization of:
  - Car movement and sensors
  - Neural network architecture
  - Training progress
  - Performance metrics

- Dynamic traffic generation with:
  - Random car placement
  - Multiple lanes
  - Variable difficulty

## Performance Metrics

The simulation tracks:
- Total reward per episode
- Completion rate
- Average speed
- Collision frequency
- Training progress

## Credits

This project builds upon the excellent foundation laid by [gniziemazity's Self-driving car project](https://github.com/gniziemazity/Self-driving-car). The original implementation provided the crucial car mechanics, visualization, and basic neural network structure that made this DQN extension possible.

