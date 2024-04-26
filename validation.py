"""
Welcome to the example of model validation with Valicy.

This example guides you through the process of validating your model with Valicy in 6 steps.

1. First, create an instance of the Valicy-API with your API key.
2. Create a system, e.g., a product.
3. Create and configure a validation scenario, e.g., at daytime.
4. Create a job for the system in the selected scenario.
5. Defining an exemplary model to show how input features from Valicy look alike and model outputs need to be presented.
6. Start the validation loop.
"""

# 1. Import and instantiate the Valicy-API with you API key
import valicy  # import the valicy API package

valicy_api = valicy.ValicyAPI(api_key='<your-api-key-here>')

# 1.1 You can change configurations for your later work. But for the beginning you don't have to change anything.
# Please see the documentation of the config method for more information.
valicy_api.config(
    number_of_instances=3,
    min_instance_predictions_before_deletion=50,
    number_of_regular_grid_points=2,
)

# 2. Before we validate anything, we need to create a "System". A system represents a product during the development process.
system: valicy.models.System = valicy_api.get_or_create_system("my_system")
# You can list all of your already created systems with: all_systems = valicy.load_all_systems()
# Additionally, you can specify more information about your system with the `system_metainfo` and `description` parameters.

# 3. Further, we need to create a "Scenario". A scenario represents a specific use case, e.g., day- vs. night time scenario.
# We need at least one scenario. You can reuse a scenario for multiple systems.
scenario = valicy_api.get_or_create_scenario("my_scenario")
# You can list all your created scenarios for this system with `all_scenarios = system.load_all_scenarios()`.
# Additionally, you can specify more information about your scenario with the `scenario_metainfo` and `scenario_description` parameters.

# 3.1 Next, we have to configure the in- and output features for the model for this scenario. This can only done once per scenario.
# As soon as the first job is created, the configuration is locked for the sake of comparability.
# Define the number of input features/dimensions and their value range for this scenario so that Valicy can forecast values from the range.
scenario.configure_features(
    valicy.models.FeatureContinuous(name="feature_0", lower=0, upper=180),
    valicy.models.FeatureContinuous(name="feature_1", lower=0, upper=180),
    valicy.models.FeatureContinuous(name="feature_2", lower=0, upper=90),
    valicy.models.FeatureContinuous(name="feature_3", lower=0, upper=90),
    valicy.models.FeatureContinuous(name="feature_4", lower=0, upper=180),
)

# Further, define the output/target/result dimensions and its truthness value of your model.
# Each output value include a threshold (limit), an orientation ("upper" or "lower") and the certainty target.
# This examples means, everything < 0.4 is considered as correct and the model should be at least 90% certain about it.
scenario.configure_output(
    valicy.models.Output(name="output_0", threshold=0.4, orientation="lower", certainty_target=0.9),
)

# 4 The next step is to create a "Job". A job represents the state of validation of your system (your product) in a scenario.
# job = valicy_api.get_or_create_job(system, scenario, "validate Product_XY_v1.0 during Daytime")
# You can list all your jobs with: `all_jobs = scenario.load_all_jobs()`.

# TODO: combine a system with a scenario to a job
job: valicy.models.Job = system + scenario + "my_job"


# 5. Here we define an exemplary model just for demonstration.
# Primarily it is about the demonstration which input values (features) Valicy provides and which output values Valicy expects.
import numpy as np  # only for our exemplary model


class ExemplaryModel:
    """This is a simple machine learning model dummy with a cosinus sinus test."""

    def validate(self, **features):
        """Inference stage of the model.
        input:
        features = {
            "feature_0":  0.0,
            "feature_1":180.0,
            "feature_2": 90.0,
            "feature_3": 90.0,
            "feature_4":  0.0,
        }
        prepare the output as expected:
            output = {"output_0": -1.0047115203588985}
        """
        target_value = 1.0  # initially set the target value to 1.0
        for feature_value in features.values():  # do something with each input feature
            cos = np.cos(feature_value / 180.0 * np.pi)
            sin = np.sin(feature_value / 180.0 * np.pi)
            random_number = 0.02 * (np.random.rand() - 0.5)
            target_value = cos + sin * target_value + random_number
            set_zero = True

            for value in features.values():  # if one feature value is between 20 and 30 or greater 40, use target value
                if (20 < value < 30) or (value > 40):
                    set_zero = False

            if set_zero:
                target_value = 0.0

        model_output = {"output_0": target_value}  # prepare the output as expected

        return model_output


model = ExemplaryModel()  # Create an instance of the exemplary model

# 6. Starting the validation loop. A "Run" represents a single validation point for your model.
# This may take some time for the first one because Valicy generate a bunch on runs.
# You can also get a run with `run = job.get_next_run()`

for i, run in enumerate(job.get_runs(number=2000)):
    # 6.1 Load the test featues for your model from a run.
    test_features = run.get_test_features()

    # 6.2 Feed the test features into your model and prepare the output.
    # 6.3 (optional) You can get a prepared result dict by `result_dict = run.get_result_dict()` where you can integrate your models output
    output = model.validate(**test_features)

    # 6.4 Send your prepared model output to Valicy.
    run.send(output)
    print(f"send data ({i}) to valicy")
