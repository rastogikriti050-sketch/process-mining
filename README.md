# Process Mining
Here goes the description of the project

## venv setup

 1. Go to your directory
 ``` 
 cd path/to/your/project

 ```

 2. Setup python venv 
 ```
 python -m venv venv
```
3. Activate it
```
venv\Scripts\activate
```

**Note**:if it shows some authentication error of some type here. Run the following commands. This is a temporary fix.

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
```
.\venv\Scripts\Activate.ps1
```
---

-> Install requirements
```
pip install -r requirements.txt
```

-> To run backend run the following command inside the venv
```
python app.py
```
