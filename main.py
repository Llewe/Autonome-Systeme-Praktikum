from codecarbon import OfflineEmissionsTracker

tracker = OfflineEmissionsTracker(output_dir="./out/", country_iso_code="DEU")
tracker.start()
# training
print("hi")

tracker.stop()