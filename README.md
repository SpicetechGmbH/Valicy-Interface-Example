# Valicy Python API Interface

A Python API wrapper for Valicy.

[[_TOC_]]


## First Steps

1. Get an API key. Please contact janis.lapins@spicetech.de
2. Import Valicy into your project.

```py
import valicy
```

3. Create a Valicy instance by providing your API key.
```py
valicy_api = valicy.Valicy("<your API key>")
```

4. Create a new `System`.
A system represents a product during the development process.

```py
system = valicy_api.get_or_create_system("Product_XY_v1.0")
```

5. Create a new `Scenario`. 
A scenario represents a specific use case, e.g., day- vs. night time scenario.
```py
scenario = valicy_api.get_or_create_scenario("Daytime")
```

5.1 Configure the input features for this scenario.
In this example, five continuous features for the model including a name and a value range.
```py
scenario.configure_features(
    valicy.models.FeatureContinuous(name="feature_0", lower=0, upper=180),
    valicy.models.FeatureContinuous(name="feature_1", lower=0, upper=180),
    valicy.models.FeatureContinuous(name="feature_2", lower=0, upper=90),
    valicy.models.FeatureContinuous(name="feature_3", lower=0, upper=90),
    valicy.models.FeatureContinuous(name="feature_4", lower=0, upper=180),
)
```

5.2 Also the output format for this scenario must be provided to Valicy.
Included is a name, the threshold and its orientation (everything < 0.4 is considered as correct) and a model certainty target.
```py
scenario.configure_output(
    valicy.models.Output(name="output_0", threshold=0.4, orientation="lower", certainty_target=0.9),
)
```

6. Create a `Job`. 
A job represents the state of validation of your system (your product) in a scenario.
```py
job: valicy.models.Job = system + scenario + "My Second Validation Job"
```
### Exemplary Model Input and Output

In order to learn about the inputs and outputs of Valicy, here is a simple model.

```py
import numpy as np
class ExemplaryModel:
    def validate(self, **features):
        target_value = 1.0  # initially set the target value to 1.0
        for feature_value in features.values():  # do something with each input feature
            cos = np.cos(feature_value / 180.0 * np.pi)
            sin = np.sin(feature_value / 180.0 * np.pi)
            random = target_value + (0.02 * (np.random.rand() - 0.5))
            target_value = cos + sin * random

            for value in features.values():  # if one feature value is between 20 and 30 or greater 40, set target to 0
                if (20 < value < 30) or (value > 40):
                    target_value = 0.0
                    break

        model_output = {"output_0": target_value}
        return model_output
model = ExemplaryModel() # create an instance of the exemplary model
```

Valicy provides you an dictionary with input features (named as configured) for your model.
```py
features = {
    "feature_0":   0.0,
    "feature_1": 180.0,
    "feature_2":  90.0,
    "feature_3":  90.0,
    "feature_4":   0.0,
}
```

Valicy expects a dictionary from you containing your output values (named as configured).

```py
output = {
    "output_0": -1.0047115203588985
}
```

### Evaluating your Model

Evaluate your model by feeding test features into it proposed by Valicy.
Each evaluation step is represented by a so-called `Run`.

1. First create a `run` from your `job` object.
    At the beginning, this may take some time because Valicy prepare multiple runs for you.
2. Fetch the `test_features`  from the `run`.
3. Feed the `test_features` to you model.
4. Send back the models output to Valicy.

```py
for run in job.get_runs(number=2000): # 1.
    test_features = run.get_test_features() # 2.
    output = model.validate(**test_features) # 3.
    run.send(output) # 4.
```

### Find the REST-API documentation here:
https://api.valicy.de/docs
