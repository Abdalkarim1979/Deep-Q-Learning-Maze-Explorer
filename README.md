# Q-Learning Maze with Neural Network  

## 📖 Project Description  
This project demonstrates a practical application of **Reinforcement Learning** using a simple maze environment. An agent learns how to reach the goal from the starting point by exploring possible moves and updating its knowledge through a Q-Learning approach, supported by a **neural network built with TensorFlow/Keras**. The training process is visualized in real time using **Pygame**, and the trained model can be saved and tested later.  

---

## ✨ Features
- 🎮 **Visual Reinforcement Learning:** Real-time visualization of the agent’s path inside the maze during training and testing.  
- 🧠 **Neural Network for Q-values:** A Keras model predicts Q-values for possible actions from the current state.  
- 🔄 **Exploration vs Exploitation:** Uses epsilon-greedy policy to balance random exploration and learned exploitation.  
- 🏆 **Reward System:** Rewards reaching the goal, penalizes invalid moves, and discourages wandering.  
- 💾 **Model Saving:** Trained model is saved in Keras format and can be reloaded for testing.  
- 🎨 **Simple GUI:** Pygame renders walls, paths, and the agent’s position clearly.  

---

## 🛠️ Technologies Used
- **Python** for core logic  
- **TensorFlow / Keras** for building and training the neural network  
- **NumPy** for state representation and matrix operations  
- **Pygame** for visualization  
- **Random / Time** for exploration and step delays  

---

## 🧠 Code Explanation
- **Maze Representation:** A 2D array where `1` = valid path, `0` = wall. Start and goal positions are defined.  
- **State Generation:** The maze is converted into a matrix with the agent’s position marked, then fed into the model.  
- **Possible Actions:** Functions check valid moves (up, down, left, right) excluding walls and boundaries.  
- **Neural Network:** Input → Flatten → Dense layers → Output of 4 Q-values (one per action).  
- **Training Loop:**  
  - Chooses actions via epsilon-greedy policy.  
  - Updates Q-values using:  
    \[
    Q \leftarrow Q + \alpha \cdot (r + \gamma \cdot \max(Q') - Q)
    \]  
  - Fits the model to approximate updated Q-values.  
  - Displays maze state and updates window caption with epoch info.  
- **Testing Loop:** Runs the trained model to navigate the maze, showing each step with a delay.  

---

## 🎯 Project Goal
The goal is to provide an **educational and interactive tool** to help learners:  
- Understand how an **order book–like maze environment** works.  
- Learn about **reinforcement learning concepts** such as states, actions, rewards, and Q-values.  
- Explore how **neural networks** can approximate Q-functions.  
- Visualize the agent’s learning process in a simple environment.  

  

---

## 🚀 How to Run  

- **Installation:** Make sure the following packages are installed: **TensorFlow**, **NumPy**, **Pygame**.  
- **Execution:** Run the file directly. Training will start for a specified number of epochs, then the model will be saved and tested.  
- **Modification:** You can change the maze, adjust learning parameters such as **γ (gamma)**, **ε (epsilon)**, and **α (alpha)**, or modify the neural network architecture to improve performance.  

---

If you need any further help or have any other questions I will be happy to help.
## Contact
  avrmicrotech@gmail.com
