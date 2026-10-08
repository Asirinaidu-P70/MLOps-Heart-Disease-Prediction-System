const form = document.querySelector("#prediction-form");
const submitButton = document.querySelector("#submit-button");
const demoButton = document.querySelector("#demo-button");
const errorMessage = document.querySelector("#form-error");
const resultSection = document.querySelector("#result");

const demoValues = {
  age: 62,
  sex: 1,
  cp: 4,
  trestbps: 170,
  chol: 300,
  fbs: 1,
  restecg: 2,
  thalach: 95,
  exang: 1,
  oldpeak: 3.8,
  slope: 3,
  ca: 3,
  thal: 7,
};

const featureNames = [
  "age",
  "sex",
  "cp",
  "trestbps",
  "chol",
  "fbs",
  "restecg",
  "thalach",
  "exang",
  "oldpeak",
  "slope",
  "ca",
  "thal",
];

demoButton.addEventListener("click", () => {
  for (const [name, value] of Object.entries(demoValues)) {
    const element = form.elements.namedItem(name);
    if (element) {
      element.value = value;
    }
  }
  form.dataset.demoMode = "true";
  form.requestSubmit();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  resultSection.hidden = true;
  submitButton.disabled = true;
  demoButton.disabled = true;
  submitButton.textContent = "Reviewing profile...";

  const features = Object.fromEntries(
    featureNames.map((name) => [name, Number(form.elements.namedItem(name).value)]),
  );

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const body = await response.json();

    if (!response.ok) {
      const detail = typeof body.detail === "string"
        ? body.detail
        : "Please check the information and try again.";
      throw new Error(detail);
    }

    const isDemo = form.dataset.demoMode === "true";
    const isHigherRisk = body.prediction === 1;

    document.querySelector("#result-title").textContent = isDemo
      ? "Example Positive Prediction"
      : isHigherRisk
        ? "A higher-risk pattern"
        : "A lower-risk pattern";
    document.querySelector("#result-message").textContent = isDemo
      ? "Model classified this fictional example as Positive"
      : body.message;
    document.querySelector("#result-note").textContent = isDemo
      ? "Fictional demonstration data — not a real patient and not a medical diagnosis."
      : body.note;
    document.querySelector("#score-value").textContent =
      `${(body.probability_of_class_1 * 100).toFixed(1)}%`;
    document.querySelector("#score-fill").style.width =
      `${Math.min(100, Math.max(0, body.probability_of_class_1 * 100))}%`;
    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    errorMessage.textContent = error.message || "The prediction could not be completed. Please try again.";
    errorMessage.hidden = false;
  } finally {
    submitButton.disabled = false;
    demoButton.disabled = false;
    submitButton.innerHTML = 'View prediction <span aria-hidden="true">&rarr;</span>';
    form.dataset.demoMode = "false";
  }
});