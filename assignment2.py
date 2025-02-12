#################################
# Your name: Ido Friedman
#################################
import math

import numpy as np
import matplotlib.pyplot as plt
import intervals as iv


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # Step 1: Generate x uniformly in [0, 1]
        xs = np.random.uniform(0, 1, m)

        # Step 2: Generate y based on P[y | x]
        ys = []
        for x in xs:
            # Determine P[y = 1 | x]
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                p_y_given_x = 0.8
            elif 0.2 < x < 0.4 or 0.6 < x < 0.8:
                p_y_given_x = 0.1
            else:
                p_y_given_x = 0  # This should not happen if x is in [0, 1]

            # Draw y from Bernoulli(p_y_given_x)
            y = np.random.choice([0, 1], p=[1 - p_y_given_x, p_y_given_x])
            ys.append(y)

        # Step 3: Return samples as a numpy array
        return np.column_stack((xs, ys))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        results = []
        for n in range(m_first, m_last + 1, step):
            empirical_errors = []
            true_errors = []

            for j in range(T):
                # Generate sample
                sample = self.sample_from_D(n)
                xs, ys = sample[:, 0], sample[:, 1]

                # Sort xs and adjust ys accordingly
                sorted_indices = np.argsort(xs)
                xs = xs[sorted_indices]
                ys = ys[sorted_indices]

                intervals, _ = iv.find_best_interval(xs, ys, k)

                # Calculate errors
                empirical_error = self.calculate_empirical_error(xs, ys, intervals)
                true_error = self.true_error(intervals)
                empirical_errors.append(empirical_error)
                true_errors.append(true_error)

            avg_empirical = np.mean(empirical_errors)
            avg_true = np.mean(true_errors)
            results.append((n, avg_empirical, avg_true))

        results = np.array(results)
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(results[:, 0], results[:, 1], label="Empirical Error", marker='o')
        plt.plot(results[:, 0], results[:, 2], label="True Error", marker='s')
        plt.xlabel("Sample Size (n)")
        plt.ylabel("Error")
        plt.title(f"Empirical vs True Error for k={k}")
        plt.legend()
        plt.grid(True)
        plt.show()

        return

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        empirical_errors = []
        true_errors = []
        k_values = []

        # Generate a sample of size m
        sample = self.sample_from_D(m)
        xs, ys = sample[:, 0], sample[:, 1]

        # Sort xs and adjust ys accordingly
        sorted_indices = np.argsort(xs)
        xs = xs[sorted_indices]
        ys = ys[sorted_indices]

        # Iterate over k values
        for k in range(k_first, k_last + 1, step):
            intervals, _ = iv.find_best_interval(xs, ys, k)

            # Calculate empirical error
            empirical_error = self.calculate_empirical_error(xs, ys, intervals)
            empirical_errors.append(empirical_error)

            # Calculate true error
            true_error = self.true_error(intervals)
            true_errors.append(true_error)

            k_values.append(k)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, empirical_errors, label="Empirical Error", marker='o')
        plt.plot(k_values, true_errors, label="True Error", marker='s')
        plt.xlabel("Number of Intervals (k)")
        plt.ylabel("Error")
        plt.title(f"Empirical and True Errors vs. Number of Intervals (n={m})")
        plt.legend()
        plt.grid(True)
        plt.show()

        return

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        empirical_errors = []
        penalties = []
        total_errors = []
        true_errors = []
        k_values = []

        # Generate a sample of size m
        sample = self.sample_from_D(m)
        xs, ys = sample[:, 0], sample[:, 1]

        # Sort xs and adjust ys accordingly
        sorted_indices = np.argsort(xs)
        xs = xs[sorted_indices]
        ys = ys[sorted_indices]

        n = m  # Number of samples
        for k in range(k_first, k_last + 1, step):
            # Find the ERM hypothesis for this k
            intervals, _ = iv.find_best_interval(xs, ys, k)

            # Calculate empirical error
            empirical_error = self.calculate_empirical_error(xs, ys, intervals)
            empirical_errors.append(empirical_error)

            # Calculate penalty
            delta_k = 0.1 / (k ** 2)
            penalty = 2 * np.sqrt((2 * k + np.log(2 / delta_k)) / n)
            penalties.append(penalty)

            # Calculate total error (empirical error + penalty)
            total_error = empirical_error + penalty
            total_errors.append(total_error)

            # Calculate true error
            true_error = self.true_error(intervals)
            true_errors.append(true_error)

            # Track the current k value
            k_values.append(k)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, empirical_errors, label="Empirical Error", marker='o')
        plt.plot(k_values, penalties, label="Penalty", marker='s')
        plt.plot(k_values, total_errors, label="Total Error (Empirical + Penalty)", marker='^')
        plt.plot(k_values, true_errors, label="True Error", marker='d')
        plt.xlabel("Number of Intervals (k)")
        plt.ylabel("Error / Penalty")
        plt.title(f"SRM: Empirical Error, Penalty, and Total Error (n={m})")
        plt.legend()
        plt.grid(True)
        plt.show()

        return

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        validation_errors = []
        k_values = []
        intervals_list = []

        # Generate a sample of size m
        sample = self.sample_from_D(m)
        xs, ys = sample[:, 0], sample[:, 1]

        # Sort xs and adjust ys accordingly
        sorted_indices = np.argsort(xs)
        xs = xs[sorted_indices]
        ys = ys[sorted_indices]

        # Split the dataset into training (80%) and validation (20%)
        train_size = int(0.8 * m)
        xs_train, ys_train = xs[:train_size], ys[:train_size]
        xs_val, ys_val = xs[train_size:], ys[train_size:]

        # Iterate over k values
        for k in range(1, 11, 1):
            # Find the ERM hypothesis for this k on the training set
            intervals, _ = iv.find_best_interval(xs_train, ys_train, k)

            # Store the intervals for later use
            intervals_list.append(intervals)

            # Calculate validation error using the new function
            validation_error = self.calculate_empirical_error(xs_val, ys_val, intervals)
            validation_errors.append(validation_error)

            # Track the current k value
            k_values.append(k)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, validation_errors, label="Validation Error", marker='o', linestyle='-')
        plt.xlabel("Number of Intervals (k)")
        plt.ylabel("Validation Error")
        plt.title(f"Validation Error vs. Number of Intervals (n={m})")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Find the best k based on validation error
        best_k_index = np.argmin(validation_errors)
        best_k = k_values[best_k_index]
        best_intervals = intervals_list[best_k_index]

        print(f"Best k value based on holdout-validation: {best_k}")
        print(f"Intervals for best k={best_k}: {best_intervals}")

        return

        #################################

    # Place for additional methods
    def true_error(self, intervals):
        """
        Calculate the true error.
        """

        def p_y_given_x(x):
            """P[y = 1 | x]"""
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                return 0.8
            elif 0.2 < x < 0.4 or 0.6 < x < 0.8:
                return 0.1
            return 0  # Should not happen if x is in [0, 1]

        # Points where P[y | x] changes
        split_points = [0, 0.2, 0.4, 0.6, 0.8, 1]

        # Initialize errors
        false_positive_error = 0
        false_negative_error = 0

        # Handle each interval
        for l, u in intervals:
            for i in range(len(split_points) - 1):
                region_l, region_u = split_points[i], split_points[i + 1]
                if l < region_u and u > region_l:
                    overlap_l = max(l, region_l)
                    overlap_u = min(u, region_u)
                    length = overlap_u - overlap_l
                    prob = 1 - p_y_given_x((overlap_l + overlap_u) / 2)
                    false_positive_error += length * prob

        # Handle complement intervals
        prev_end = 0
        for l, u in intervals:
            if l > prev_end:
                false_negative_error += (l - prev_end) * p_y_given_x((prev_end + l) / 2)
            prev_end = max(prev_end, u)
        if prev_end < 1:
            false_negative_error += (1 - prev_end) * p_y_given_x((prev_end + 1) / 2)

        # Total error
        return false_positive_error + false_negative_error
        #################################

    def calculate_empirical_error(self, xs, ys, intervals):
        # Calculate predictions based on the intervals
        predicted_labels = [1 if any(l <= x <= u for l, u in intervals) else 0 for x in xs]

        # Calculate the proportion of incorrect predictions
        incorrect_predictions = sum(y != pred for y, pred in zip(ys, predicted_labels))
        empirical_error = incorrect_predictions / len(ys)

        return empirical_error


if __name__ == '__main__':
    ass = Assignment2()
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    #ass.cross_validation(1500)
