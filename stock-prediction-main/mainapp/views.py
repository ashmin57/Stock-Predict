from django.shortcuts import render
from django.conf import settings
import subprocess
import os
import time
import statistics
import csv
from django.http import JsonResponse
from django.http import HttpResponse
from .forms import CSVUploadForm
import plotly.graph_objs as go
from plotly.offline import plot
from selenium import webdriver
from .arima import arima_model
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from .models import News  # Add this line to import the News model
from django.contrib.admin.views.decorators import staff_member_required


@staff_member_required
def admin_dashboard(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Handle CSV upload and training logic here
            messages.success(request, 'CSV uploaded and model trained successfully!')
            return redirect('admin_dashboard')
    else:
        form = CSVUploadForm()

    return render(request, 'admin_dashboard.html', {'form': form})

def auto_download(request):
    if request.method == 'POST':
        company = request.POST.get('company')

        brave_path = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"  # Update this path if different
        chrome_driver_path = 'D:\\stock\\arima\\chromedriver\\chromedriver.exe'

        chrome_options = Options()
        chrome_options.binary_location = brave_path
        chrome_options.add_argument("--headless")  # Update if you want to use headless browser

        service = Service(chrome_driver_path)
        driver = None

        try:
            print("Navigating to the page...")
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get('https://nepsealpha.com/nepse-data')

            wait = WebDriverWait(driver, 10)

            print("Waiting for select element...")
            select_click = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#vue_app_content > div.page.page_margin_top > div > div > div > form > div > div > div:nth-child(4) > span > span.selection > span')))
            select_click.click()
            print("Select element clicked.")

            print("Entering start date...")
            start_date = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#vue_app_content > div.page.page_margin_top > div > div > div > form > div > div > div:nth-child(2) > input')))
            start_date.send_keys("07/01/2013")
            print("Start date entered.")

            print("Searching for the company...")
            select_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'body > span > span > span.select2-search.select2-search--dropdown > input')))
            select_input.send_keys(company)
            select_input.send_keys(Keys.ENTER)
            print("Company selected.")

            print("Clicking filter button...")
            filter_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#vue_app_content > div.page.page_margin_top > div > div > div > form > div > div > div:nth-child(5) > button')))
            filter_button.click()
            print("Filter button clicked.")

            time.sleep(3)  # Adjust as needed for file download to complete

            print("Waiting for CSV button...")
            csv_button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#result-table_wrapper > div.dt-buttons > button.dt-button.buttons-csv.buttons-html5.btn.btn-outline-secondary.btn-sm')))
            csv_button.click()
            print("CSV button clicked.")

            # Allow time for file download to complete
            time.sleep(20)  # Adjust as needed for file download to complete

        except TimeoutException as e:
            print(f"TimeoutException: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if driver:
                driver.quit()

        # Check for downloaded files and open download folder
        download_folder = os.path.expanduser("D:\\stock\\arima\\nepse")
        downloaded_files = os.listdir(download_folder)

        if downloaded_files:
            print(f"Downloaded files: {downloaded_files}")
        else:
            print("No files were downloaded.")

        # Open download folder if file(s) found
        subprocess.Popen(f'explorer "{download_folder}"')

        return render(request, 'data.html')

import torch
import pandas as pd
from .utils import preprocess_csv  # Assuming you have a function to preprocess the CSV data

def predict(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        model_choice = request.POST.get('model')
        csv_file = request.FILES['csv_file']
        
        # Load and preprocess the CSV file
        preprocessed_data = preprocess_csv(csv_file)  # Convert CSV into format suitable for LSTM model
        
        if model == 'LSTM':
            from .lstm import lstm_model
            result = lstm_model(preprocessed_data)
        
        return JsonResponse({'data': result_dict})
    
    return render(request, 'predict.html')



def data_download(request):
    return render(request, 'data.html')


import csv
import statistics
from datetime import datetime
import plotly.graph_objects as go
from plotly.offline import plot
from django.shortcuts import render
from .forms import CSVUploadForm  # Adjust the import based on your project structure

def visualize_csv_form(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            reader = csv.reader(csv_file.read().decode('utf-8').splitlines())
            header = next(reader)  # Skip the header row
            data = list(reader)

            # Extract and parse column data
            dates = [datetime.strptime(row[1], '%m/%d/%Y') for row in data]  # Assuming the date column is at index 1
            close_prices = [float(row[5]) for row in data]  # Assuming the close price column is at index 5

            # Combine and sort data by date
            combined_data = sorted(zip(dates, close_prices), key=lambda x: x[0])
            sorted_dates, sorted_close_prices = zip(*combined_data)

            # Calculate statistical data
            minimum = min(sorted_close_prices)
            maximum = max(sorted_close_prices)
            average = statistics.mean(sorted_close_prices)
            variance = statistics.variance(sorted_close_prices)
            median = statistics.median(sorted_close_prices)

            # Prepare Plotly chart
            chart_data = go.Scatter(x=sorted_dates, y=sorted_close_prices, mode='lines', name='Close Prices')
            layout = go.Layout(title='Close Prices Over Time', xaxis=dict(title='Date'), yaxis=dict(title='Close Price'))
            fig = go.Figure(data=[chart_data], layout=layout)
            plot_div = plot(fig, output_type='div')

            return render(request, 'visualization.html', {
                'form': form,
                'plot_div': plot_div,
                'minimum': minimum,
                'maximum': maximum,
                'average': average,
                'variance': variance,
                'median': median
            })
    else:
        form = CSVUploadForm()

    return render(request, 'visualization.html', {'form': form})



def get_driver():
    brave_path = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"  
    chrome_options = Options()
    chrome_options.binary_location = brave_path
    chrome_options.add_argument("--headless")
    service = Service('D:\\stock\\arima\\chromedriver\\chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def index(request):
    return render(request, 'index.html')


from django.shortcuts import render
from django.utils.timezone import now
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from .models import News

def get_driver():
    # Ensure to replace this function with actual driver initialization code
    from selenium import webdriver
    driver = webdriver.Chrome()
    return driver

def news(request):
    import time
    ts = time.time()  # Get current timestamp
    
    try:
        # Fetch the latest expiry time from the database
        db_exp_time = News.objects.values('expiry').latest('id')['expiry']
        if ts < db_exp_time:  # If current time is before expiry, serve cached news
            db_data = News.objects.all().order_by('id').values()
            return render(request, 'news.html', {'news': db_data})
        else:
            # Scrape new news data if cache is expired
            driver = get_driver()

            try:
                # Load the news page
                driver.get('https://merolagani.com/NewsList.aspx')
                WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "#ctl00_ContentPlaceHolder1_divData > .btn-block"))
                ).click()
                time.sleep(2)

                # Extract news data
                img = driver.find_elements(By.CSS_SELECTOR, '.media-wrap > a > img')
                img_data = [i.get_attribute('src') for i in img]

                hrefs = driver.find_elements(By.CSS_SELECTOR, '.media-wrap > a')
                single_news_href_data = [i.get_attribute('href') for i in hrefs]

                news_link = driver.find_elements(By.CLASS_NAME, 'media-body')
                news_titledate_data = [i.text.replace("\n", "<br>") for i in news_link]

                # Prepare news data
                news_data = [{'title': news_titledate_data[i], 'link': single_news_href_data[i], 'image': img_data[i]} for i in range(len(news_titledate_data))]

            except Exception as e:
                print(f"Error during web scraping: {e}")
                driver.quit()
                return render(request, 'news.html', {'news': None})

            finally:
                driver.quit()

            # If the correct amount of news is found, update the database
            if len(news_data) == 16:
                expiry_time = ts + 9000  # Set expiry time (2.5 hours)

                # Clear old news and insert new news
                News.objects.all().delete()
                for i in news_data:
                    add_news = News(title=i['title'], image=i['image'], link=i['link'], expiry=expiry_time)
                    add_news.save()

                # Fetch fresh data from the database
                db_data = News.objects.all().order_by('id').values()
                return render(request, 'news.html', {'news': db_data})
            else:
                # If less than 16 articles were scraped, display an error message
                return render(request, 'news.html', {'news': None})
    except News.DoesNotExist:
        # Handle case where there is no news data in the database yet
        print("No news in the database, scraping for the first time.")
        return refresh_news_data(request)
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"Unexpected error: {e}")
        return render(request, 'news.html', {'news': None})


def refresh_news_data(request):
    """
    This function handles refreshing the news data from the website 
    in case of the first-time fetch or cache expiry.
    """
    driver = get_driver()
    try:
        driver.get('https://merolagani.com/NewsList.aspx/')
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#ctl00_ContentPlaceHolder1_divData > .btn-block"))
        ).click()
        time.sleep(2)

        # Extract news data
        img = driver.find_elements(By.CSS_SELECTOR, '.media-wrap > a > img')
        img_data = [i.get_attribute('src') for i in img]

        hrefs = driver.find_elements(By.CSS_SELECTOR, '.media-wrap > a')
        single_news_href_data = [i.get_attribute('href') for i in hrefs]

        news_link = driver.find_elements(By.CLASS_NAME, 'media-body')
        news_titledate_data = [i.text.replace("\n", "<br>") for i in news_link]

        news_data = [{'title': news_titledate_data[i], 'link': single_news_href_data[i], 'image': img_data[i]} for i in range(len(news_titledate_data))]

    except Exception as e:
        print(f"Error during web scraping: {e}")
        return render(request, 'news.html', {'news': None})

    finally:
        driver.quit()

    # If the correct amount of news is found, update the database
    if len(news_data) == 16:
        expiry_time = time.time() + 9000  # Set expiry time

        # Clear old news and insert new news
        News.objects.all().delete()
        for i in news_data:
            add_news = News(title=i['title'], image=i['image'], link=i['link'], expiry=expiry_time)
            add_news.save()

        # Fetch fresh data from the database
        db_data = News.objects.all().order_by('id').values()
        return render(request, 'news.html', {'news': db_data})
    else:
        # If less than 16 articles were scraped, display an error message
        return render(request, 'news.html', {'news': None})