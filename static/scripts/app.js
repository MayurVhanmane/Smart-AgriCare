document.addEventListener("DOMContentLoaded", function () {
  var languageForm = document.querySelector("[data-language-form]");
  var languageSelect = document.querySelector("[data-language-select]");

  if (languageForm && languageSelect) {
    languageSelect.addEventListener("change", function () {
      languageForm.submit();
    });
  }

  var navToggle = document.querySelector("[data-nav-toggle]");
  var navLinks = document.querySelector("[data-nav-links]");

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      var isOpen = navLinks.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(isOpen));
    });
  }

  document.querySelectorAll("[data-progress-form]").forEach(function (form) {
    var inputs = Array.from(form.querySelectorAll("input[required], select[required]"));
    var formCard = form.closest(".form-card") || form;
    var progress = formCard.querySelector("[data-progress]");
    var label = formCard.querySelector("[data-progress-label]");

    function updateProgress() {
      var completed = inputs.filter(function (input) {
        return input.value.trim() !== "";
      }).length;
      var percent = inputs.length ? Math.round((completed / inputs.length) * 100) : 0;
      if (progress) progress.style.width = percent + "%";
      if (label) {
        var readyText = label.dataset.progressReady || "Ready to analyze";
        var progressFormat = label.dataset.progressFormat || "{completed} of {total} completed";
        label.textContent = percent === 100
          ? readyText
          : progressFormat
              .replace("{completed}", String(completed))
              .replace("{total}", String(inputs.length));
      }
      var track = formCard.querySelector('[role="progressbar"]');
      if (track) track.setAttribute("aria-valuenow", String(percent));
    }

    form.addEventListener("input", updateProgress);
    form.addEventListener("change", updateProgress);
    form.addEventListener("focusout", updateProgress);
    updateProgress();
  });

  var fileInput = document.querySelector("[data-image-input]");
  var uploadZone = document.querySelector("[data-upload-zone]");
  var preview = document.querySelector("[data-image-preview]");
  var previewImage = document.querySelector("[data-preview-image]");
  var previewName = document.querySelector("[data-preview-name]");

  function showImage(file) {
    if (!file || !file.type.startsWith("image/")) return;
    var reader = new FileReader();
    reader.addEventListener("load", function () {
      if (previewImage && typeof reader.result === "string") previewImage.src = reader.result;
      if (previewName) previewName.textContent = file.name;
      if (preview) preview.classList.add("is-visible");
    });
    reader.readAsDataURL(file);
  }

  if (fileInput) {
    fileInput.addEventListener("change", function () {
      showImage(fileInput.files && fileInput.files[0]);
    });
  }

  if (uploadZone) {
    ["dragenter", "dragover"].forEach(function (eventName) {
      uploadZone.addEventListener(eventName, function (event) {
        event.preventDefault();
        uploadZone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach(function (eventName) {
      uploadZone.addEventListener(eventName, function (event) {
        event.preventDefault();
        uploadZone.classList.remove("is-dragging");
      });
    });
    uploadZone.addEventListener("drop", function (event) {
      var file = event.dataTransfer && event.dataTransfer.files[0];
      if (file && fileInput) {
        fileInput.files = event.dataTransfer.files;
        showImage(file);
      }
    });
  }
});
