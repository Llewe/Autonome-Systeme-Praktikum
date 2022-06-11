

from codecarbon import OfflineEmissionsTracker


print("hi")


tracker = OfflineEmissionsTracker(country_iso_code="DEU")
tracker.start()
# training
print("hi")

tracker.stop()