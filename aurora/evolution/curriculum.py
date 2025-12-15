import numpy as np

class CurriculumManager:
    def __init__(self):
        self.levels = [
            {'name': 'Basic', 'difficulty': 1},
            {'name': 'Intermediate', 'difficulty': 3},
            {'name': 'Advanced', 'difficulty': 5},
            {'name': 'Expert', 'difficulty': 7},
            {'name': 'Master', 'difficulty': 10}
        ]
        self.current_level_idx = 0
        self.history_window = 10
        self.performance_history = []
        self.success_threshold = 0.75 # Advance if avg reward > 0.75

    def get_current_difficulty(self):
        return self.levels[self.current_level_idx]['difficulty']
    
    def get_current_level_name(self):
        return self.levels[self.current_level_idx]['name']

    def record_performance(self, reward):
        self.performance_history.append(reward)
        if len(self.performance_history) > self.history_window:
            self.performance_history.pop(0)

    def check_progression(self):
        """
        Check if we should advance to the next level.
        Returns True if advanced, False otherwise.
        """
        if self.current_level_idx >= len(self.levels) - 1:
            return False # Max level

        if len(self.performance_history) < 5:
            return False # Need more data

        avg_perf = np.mean(self.performance_history)
        if avg_perf >= self.success_threshold:
            self.current_level_idx += 1
            self.performance_history = [] # Reset history for new level
            return True
            
        return False
