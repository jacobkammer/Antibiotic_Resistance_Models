# Below represents the world that the vacuum cleaner lives in
#creates a two room world 
class VacuumEnvironment:
    def __init__(self, dirt=None):
        # dirt: dict (location -> dirty: bool)
        # e.g., {"A": True, "B": False}
        if dirt is None:#parameter represening the initial state of the environment
            dirt = {"A": False, "B": False}#default state of no dirt
        self.dirt = dirt.copy()
        self.agent_location = "A"#always start at state A

# Sensor that gathers information about current state of environment
class Sensor:
    def perceive(self, env: VacuumEnvironment):
        """ first obtain current location and
        then determine if there is dirt at 
        given location.  The code with return a tuple 
        with the information A or B and dirt or no dirt"""
        return env.agent_location, env.dirt[env.agent_location]


#Actuator part of agent.  It executes octions that will change the environmental state
#there are two parameters: env and action
class Actuator:
    def act(self, env: VacuumEnvironment, action: str):
        """3 defined actions, suck, move left
        or move right"""
        if action == "Suck":
            env.dirt[env.agent_location] = False
        elif action == "Left":
            env.agent_location = "A"
        elif action == "Right":
            env.agent_location = "B"

# The code below defines the VacuumAgent and uses sensor and actuator 
#that were define above
class VacuumAgent:
    def __init__(self, sensor: Sensor, actuator: Actuator):
        self.sensor = sensor
        self.actuator = actuator

    def decide(self, percept):
        """this method implements the decision making logic
        of the agent.  Input is a precept which contains location 
        and the state of environment clean or dirty. clean dirt when
        found if no dirt move to other room"""
        location, dirty = percept
        if dirty: #if dirty then clean, this is the primary function
            return "Suck"
        if location == "A":#if location A and no dirt (you just cleaned it up), move right to position B
            return "Right"
        else:#if location B and no dirt (you just cleaned it up) move left
            return "Left"

    def run_with_performance(self, env: VacuumEnvironment, percept_sequence, performance_measure, performance_measure2):
        """executes agent through percepts with performance tracking"""
        print("Action Sequence and Performance:")
        print("-" * 50)
        
        for i, percept in enumerate(percept_sequence):
            location, dirty = percept
            print(f"Step {i+1}: Percept: {percept}")
            
            action = self.decide(percept)
            print(f"  Action: {action}")
            
            # Check if location was dirty before action (for scoring)
            was_dirty = env.dirt[location]
            
            # Execute action
            self.actuator.act(env, action)
            
            # Update performance measure
            if action == "Suck" and was_dirty:
                performance_measure.add_point(location)
                print(f"  Cleaned {location}! +1 point")
            
            # Update second performance measure
            step_score = performance_measure2.calculate_step_score(env, action)
            print(f"  Step score: {step_score}")
            print(f"  Score after action: {performance_measure.get_score()} (measure 1), {performance_measure2.get_score()} (measure 2)")
            print(f"  Environment: dirt={env.dirt}, location={env.agent_location}")
            print()
        
        return performance_measure.get_score(), performance_measure2.get_score()

# Performance Measure Class that awards points for cleaning a location
class PerformanceMeasure:
    def __init__(self):
        self.score = 0
        self.cleaned_locations = []
    
    def add_point(self, location):
        """Add one point for cleaning a location"""
        self.score += 1
        self.cleaned_locations.append(location)#records which locations have been cleaned
    #list that keeps track of cleaned locations
#the following two methods are getter methods that provide access to 
#score and cleaned locations
    def get_score(self):#method definition
        """Get current score"""
        return self.score#returns the current score
    
    def get_cleaned_locations(self):
        """Get list of cleaned locations"""
        return self.cleaned_locations
 #method below returns performance measure to initial state.
 # #starts fresh between different test sequences   
    def reset(self):
        """Reset the performance measure"""
        self.score = 0#set back to 0
        self.cleaned_locations = []#reset cleaned locations

# Second Performance Measure Class
class PerformanceMeasure2:
    def __init__(self):
        self.score = 0
        self.step_details = []
    
    def calculate_step_score(self, env: VacuumEnvironment, action: str):
        """Calculate score for this time step: +1 for each clean square, -1 for moves"""
        # Count clean squares (locations that are False in dirt dict)
        clean_squares = sum(1 for dirty in env.dirt.values() if not dirty)
        
        # Penalty for movement
        move_penalty = -1 if action in ["Left", "Right"] else 0
        
        step_score = clean_squares + move_penalty
        self.score += step_score
        
        self.step_details.append({
            'clean_squares': clean_squares,
            'move_penalty': move_penalty,
            'step_score': step_score,
            'total_score': self.score
        })
        
        return step_score
   #the following two methods are getter methods that provide access to 
   #score and step details
    def get_score(self):
        """Get current total score"""
        return self.score
    
    def get_step_details(self):
        """Get detailed breakdown of each step"""
        return self.step_details
    
    def reset(self):
        """Reset the performance measure"""
        self.score = 0
        self.step_details = []

#Code below creates an instance of the the needed components to run the vacuum cleaner simulation.  This is the setup phase
#Create environment
env = VacuumEnvironment({"A": True, "B": True})  # Both A and B locations start dirty
#create sensor part that will observe environment and get percepts (location + dirt status)
sensor = Sensor()

#create actuator part that will modify environment. it executes actions
actuator = Actuator()

#creates main agent giving it a sensor and actuator
agent = VacuumAgent(sensor, actuator)#agent can sense and act

# Create performance measures
performance_measure = PerformanceMeasure()
performance_measure2 = PerformanceMeasure2()

# Sequence 1: [A, Dirty], [B, Dirty] repeated 50 times
sequence_1 = [("A", True), ("B", True)] * 50

# Sequence 2: [A, Dirty], [B, Dirty] once, followed by [A, Clean], [B, Clean] for 49 repetitions
sequence_2 = [("A", True), ("B", True)] + [("A", False), ("B", False)] * 49

# Run the simulations
print("=== Vacuum Cleaner Simulation ===")
print(f"Initial environment: dirt={env.dirt}, location={env.agent_location}")
print()

# === SEQUENCE 1 ===
print("SEQUENCE 1: [A, Dirty], [B, Dirty] repeated 50 times")
print("(Showing first 10 steps only for brevity)")
print()
final_score1, final_score2 = agent.run_with_performance(env, sequence_1[:10], performance_measure, performance_measure2)
print("SEQUENCE 1 RESULTS:")
print(f"  Final Score (Measure 1 - Cleaning Points): {final_score1}")
print(f"  Final Score (Measure 2 - Clean Squares + Move Penalty): {final_score2}")
print()

print("=" * 60)
print()

# === SEQUENCE 2 ===
# Reset environment and performance measures for second sequence
env = VacuumEnvironment({"A": True, "B": True})
performance_measure.reset()
performance_measure2.reset()

print("SEQUENCE 2: [A, Dirty], [B, Dirty] once, then [A, Clean], [B, Clean] for 49 repetitions")
print("(Showing first 10 steps only for brevity)")
print()
final_score1, final_score2 = agent.run_with_performance(env, sequence_2[:10], performance_measure, performance_measure2)
print("SEQUENCE 2 RESULTS:")
print(f"  Final Score (Measure 1 - Cleaning Points): {final_score1}")
print(f"  Final Score (Measure 2 - Clean Squares + Move Penalty): {final_score2}")
print()

print("=" * 60)
print("SIMULATION SUMMARY:")
print(f"  Total steps in Sequence 1: {len(sequence_1)}")
print(f"  Total steps in Sequence 2: {len(sequence_2)}")
print("=" * 60)