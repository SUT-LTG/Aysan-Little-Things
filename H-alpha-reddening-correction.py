import numpy as np
import pandas as pd

# Fixed internal reddening for all regions
internal_reddening = 0.1

# Function to calculate extinction using the CCM law
def ccm_extinction(lam, E_BV, R_V=3.086):
    """
    Calculate the extinction A(lambda) at a given wavelength (in microns)
    using the Cardelli-Clayton-Mathis (CCM) extinction law.
    """
    # Convert wavelength (lam) to inverse microns (x)
    x = 1.0 / lam

    # Define y as in the CCM formulation
    y = x - 1.82

    # Calculate the coefficients a(x) and b(x) using the CCM optical polynomials
    a = (1 +
         0.17699 * y -
         0.50447 * y**2 -
         0.02427 * y**3 +
         0.72085 * y**4 +
         0.01979 * y**5 -
         0.77530 * y**6 +
         0.32999 * y**7)

    b = (1.41338 * y +
         2.28305 * y**2 +
         1.07233 * y**3 -
         5.38434 * y**4 -
         0.62251 * y**5 +
         5.30260 * y**6 -
         2.09002 * y**7)

    # Compute A(V) from E(B-V) using the relation A(V) = R_V * E(B-V)
    A_V = R_V * E_BV

    # Compute the extinction at the given wavelength using:
    # A(lambda) = [a(x) + b(x)/R_V] * A(V)
    A_lambda = (a + b / R_V) * A_V

    return A_lambda

# Function to correct flux for reddening
def correct_flux(F_obs, A_lambda):
    """
    Correct the observed flux for reddening using A(lambda).
    """
    F_intr = F_obs * 10**(0.4 * A_lambda)
    return F_intr

# Main script
if __name__ == '__main__':
    # Read the CSV file
    filepath = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions\H-alpha analysis.csv"
    data = pd.read_csv(filepath)

    # Define the wavelength for Hα (in microns)
    lam_Ha = 0.6563  # Hα wavelength in microns

    # Compute the total reddening (foreground + internal)
    data['Total_E(B-V)'] = data['E(B-V)'] + internal_reddening

    # Calculate A(lambda) for each region
    data['A_lambda'] = data['Total_E(B-V)'].apply(lambda E_BV: ccm_extinction(lam_Ha, E_BV))

    # Correct the observed flux (column 5: Bootstrap with Replacement Mean)
    data['Intrinsic Flux'] = data.apply(lambda row: correct_flux(row['Bootstrap with Replacement Mean'], row['A_lambda']), axis=1)

    # Save the updated DataFrame to a new CSV file
    output_filepath = r"C:\Users\AYSAN\Desktop\project\Galaxy\H-alpha regions\Corrected_H-alpha_analysis.csv"
    data.to_csv(output_filepath, index=False)

    print(f"Reddening correction applied. Saved to {output_filepath}")
