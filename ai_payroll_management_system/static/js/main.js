// PayPilot AI Client-Side Scripts
document.addEventListener("DOMContentLoaded", function () {
  // Auto-dismiss flash alerts after 6 seconds
  setTimeout(function () {
    const alerts = document.querySelectorAll(".alert-dismissible");
    alerts.forEach(function (alert) {
      const bsAlert = bootstrap.Alert.getInstance(alert);
      if (bsAlert) {
        bsAlert.close();
      }
    });
  }, 6000);
});
