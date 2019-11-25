# Generate the fake data
python make_fake_data.py

# Run the single pulse inference for pulse number 0 with 3 shapelets
frodo_single_pulse -p 0 -s 3 -d fake_data.txt --plot
