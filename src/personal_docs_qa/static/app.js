const form = document.querySelector(".question-form");
const answer = document.querySelector("#answer");

form?.addEventListener("submit", (event) => {
  event.preventDefault();
  answer.textContent = "The web shell is ready. The shared QA engine is not implemented yet.";
});

