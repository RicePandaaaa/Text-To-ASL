import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self, model, fps):
        # Graph info
        self.NUM_POINTS = 21
        self.num_frames = max(model["frame"])
        self.model = model
        self.fps = fps

        # Hand data
        self.left_hand_landmarks_all_frames = self.model[self.model["type"] == "left_hand"].groupby("frame")
        self.right_hand_landmarks_all_frames = self.model[self.model["type"] == "right_hand"].groupby("frame")

        # Set up the plot and scatter
        self.fig, self.ax_left, self.ax_right, self.left_scatter, self.right_scatter = self.setup_plot()

        # Store the connected landmark pairs
        self.connected_landmarks = [
            [(0, 1), (1, 2), (2, 3), (3, 4)],              # Thumb
            [(0, 5), (5, 9), (5, 13), (13, 17), (0, 17)],   # Palm
            [(5, 6), (6, 7), (7, 8)],                      # Index Finger
            [(9, 10), (10, 11), (11, 12)],                  # Middle Finger
            [(13, 14), (14, 15), (15, 16)],                 # Ring Finger
            [(17, 18), (17, 18), (18, 19), (19, 20)]        # Pinky Finger
        ]

    def setup_plot(self):
        """
        Set up the 3D plots
        """
        # Set up 3D plots with subplots (2 columns: one for left hand, one for right hand)
        fig = plt.figure(figsize=(10, 5))

        # Make mins and maxes for the dimensions
        x_min, x_max = self.model["x"].min(), self.model["x"].max()
        y_min, y_max = self.model["y"].min(), self.model["y"].max()
        z_min, z_max = self.model["z"].min(), self.model["z"].max()

        # Left hand subplot
        ax_left = fig.add_subplot(121, projection="3d")
        ax_left.set_title("Left Hand")
        ax_left.set_xlim(x_min, x_max)
        ax_left.set_ylim(y_min, y_max)
        ax_left.set_zlim(z_min, z_max)
        ax_left.set_xlabel("X")
        ax_left.set_ylabel("Y")
        ax_left.set_zlabel("Z")

        # Right hand subplot
        ax_right = fig.add_subplot(122, projection="3d")
        ax_right.set_title("Right Hand")
        ax_right.set_xlim(x_min, x_max)
        ax_right.set_ylim(y_min, y_max)
        ax_right.set_zlim(z_min, z_max)
        ax_right.set_xlabel("X")
        ax_right.set_ylabel("Y")
        ax_right.set_zlabel("Z")

        # Initialize scatter plots for left and right hand (empty initially)
        left_scatter = ax_left.scatter([], [], [], c="blue", marker="o", label="Left Hand")
        right_scatter = ax_right.scatter([], [], [], c="red", marker="o", label="Right Hand")

        return fig, ax_left, ax_right, left_scatter, right_scatter

    def init_animation(self):
        # Initialize the scatter plots to be empty
        self.left_scatter._offsets3d = ([], [], [])
        self.right_scatter._offsets3d = ([], [], [])
        return self.left_scatter, self.right_scatter

    def update_animation(self, frame):
        """
        Remake the plots with the new points
        """
        # Clear previous frames and re-apply labels
        self.clear_axes()

        # Update the left hand plot for the current frame
        self.update_hand_plot(self.ax_left, self.left_hand_landmarks_all_frames, self.left_scatter, frame, self.connected_landmarks, "blue")

        # Update the right hand plot for the current frame
        self.update_hand_plot(self.ax_right, self.right_hand_landmarks_all_frames, self.right_scatter, frame, self.connected_landmarks, "red")

        return self.left_scatter, self.right_scatter

    def clear_axes(self):
        """
        Clear and reset the axis labels for both left and right hand plots
        """
        self.ax_left.clear()
        self.ax_left.set_xlabel("X")
        self.ax_left.set_ylabel("Y")
        self.ax_left.set_zlabel("Z")
        self.ax_left.set_title("Left Hand")

        self.ax_right.clear()
        self.ax_right.set_xlabel("X")
        self.ax_right.set_ylabel("Y")
        self.ax_right.set_zlabel("Z")
        self.ax_right.set_title("Right Hand")

    def update_hand_plot(self, ax, hand_data, scatter, frame, connected_landmarks, color):
        """
        Update the hand plot with new data for the given frame
        """
        if frame in hand_data.groups:
            data = hand_data.get_group(frame)

            # Update scatter plot
            scatter._offsets3d = (data["x"].values, data["y"].values, data["z"].values)
            ax.scatter(data["x"], data["y"], data["z"], c=color)

            # Plot lines for connected landmark pairs
            color_strength = 0.2
            for pair in connected_landmarks:
                for i, j in pair:
                    ax.plot(
                        [data["x"].values[i], data["x"].values[j]],
                        [data["y"].values[i], data["y"].values[j]],
                        [data["z"].values[i], data["z"].values[j]],
                        c=(0, 0, color_strength) if color == "blue" else (color_strength, 0, 0)
                    )
                color_strength += 0.15

            # Dynamically update axis limits
            ax.set_xlim(data["x"].min(), data["x"].max())
            ax.set_ylim(data["y"].min(), data["y"].max())
            ax.set_zlim(data["z"].min(), data["z"].max())
        else:
            scatter._offsets3d = ([], [], [])

    def start_animation(self):
        """
        Starts the actual animation
        """
        # Set the initial view of both plots
        self.ax_left.view_init(elev=-24, azim=-170, roll=90)
        self.ax_right.view_init(elev=-156, azim=170, roll=90)

        # Create the animation
        anim = FuncAnimation(self.fig, self.update_animation, frames=range(self.num_frames + 1), init_func=self.init_animation, interval=1000/self.fps, blit=False)

        # Display the animation
        plt.show()


class SinglePlotVisualizer:
    def __init__(self, model, fps):
        # Graph info
        self.NUM_POINTS = 21
        self.num_frames = max(model["frame"])
        self.model = model
        self.fps = fps

        # Hand data
        self.left_hand_landmarks_all_frames = self.model[self.model["type"] == "left_hand"].groupby("frame")
        self.right_hand_landmarks_all_frames = self.model[self.model["type"] == "right_hand"].groupby("frame")

        # Graph dimensions
        self.x_min, self.x_max = self.model["x"].min(), self.model["x"].max()
        self.y_min, self.y_max = self.model["y"].min(), self.model["y"].max()
        self.z_min, self.z_max = self.model["z"].min(), self.model["z"].max()

        # Set up the plot and scatter
        self.fig, self.ax, self.left_scatter, self.right_scatter = self.setup_plot()

        # Store the connected landmark pairs
        self.connected_landmarks = [
            [(0, 1), (1, 2), (2, 3), (3, 4)],              # Thumb
            [(0, 5), (5, 9), (5, 13), (13, 17), (0, 17)],   # Palm
            [(5, 6), (6, 7), (7, 8)],                      # Index Finger
            [(9, 10), (10, 11), (11, 12)],                  # Middle Finger
            [(13, 14), (14, 15), (15, 16)],                 # Ring Finger
            [(17, 18), (17, 18), (18, 19), (19, 20)]        # Pinky Finger
        ]

    def setup_plot(self):
        """
        Set up the 3D plot for both hands
        """
        # Set up 3D plot
        fig = plt.figure(figsize=(8, 8))

        # Single subplot for both hands
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Hand Landmarks")
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.z_min, self.z_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Initialize scatter plots for both hands (empty initially)
        left_scatter = ax.scatter([], [], [], c="blue", marker="o", label="Left Hand")
        right_scatter = ax.scatter([], [], [], c="red", marker="o", label="Right Hand")

        return fig, ax, left_scatter, right_scatter

    def init_animation(self):
        # Initialize the scatter plots to be empty
        self.left_scatter._offsets3d = ([], [], [])
        self.right_scatter._offsets3d = ([], [], [])
        return self.left_scatter, self.right_scatter

    def update_animation(self, frame):
        """
        Remake the plots with the new points
        """
        # Clear previous frames and re-apply labels
        self.clear_axes()

        # Update the left hand plot for the current frame
        self.update_hand_plot(self.left_hand_landmarks_all_frames, self.left_scatter, frame, self.connected_landmarks, "blue")

        # Update the right hand plot for the current frame
        self.update_hand_plot(self.right_hand_landmarks_all_frames, self.right_scatter, frame, self.connected_landmarks, "red")

        return self.left_scatter, self.right_scatter

    def clear_axes(self):
        """
        Clear and reset the axis labels for the single plot
        """
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Hand Landmarks")

    def update_hand_plot(self, hand_data, scatter, frame, connected_landmarks, color):
        """
        Update the hand plot with new data for the given frame
        """
        if frame in hand_data.groups:
            data = hand_data.get_group(frame)

            # Update scatter plot
            scatter._offsets3d = (data["x"].values, data["y"].values, data["z"].values)
            self.ax.scatter(data["x"], data["y"], data["z"], c=color)

            # Plot lines for connected landmark pairs
            color_strength = 0.2
            for pair in connected_landmarks:
                for i, j in pair:
                    self.ax.plot(
                        [data["x"].values[i], data["x"].values[j]],
                        [data["y"].values[i], data["y"].values[j]],
                        [data["z"].values[i], data["z"].values[j]],
                        c=(0, 0, color_strength) if color == "blue" else (color_strength, 0, 0)
                    )
                color_strength += 0.15

            self.ax.set_xlim(self.x_min, self.x_max)
            self.ax.set_ylim(self.y_min, self.y_max)
            self.ax.set_zlim(self.z_min, self.z_max)

        else:
            scatter._offsets3d = ([], [], [])

    def start_animation(self):
        """
        Starts the actual animation
        """
        # Set the initial view of the plot
        self.ax.view_init(elev=-24, azim=-170, roll=90)

        # Create the animation
        anim = FuncAnimation(self.fig, self.update_animation, frames=range(self.num_frames + 1), init_func=self.init_animation, interval=1000/self.fps, blit=False)

        # Display the animation
        plt.show()

