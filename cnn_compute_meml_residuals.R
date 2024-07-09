## Compute MEML residuals 
#### Parameters ####

location_name <- 'SFBA'

min_evs <- 10
min_stns <- 10

dir.create(sprintf("/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/%devs_%dstns/",min_evs,min_stns))
dir.create(sprintf("/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/%devs_%dstns/R_output/",min_evs,min_stns))
dir.create(sprintf("/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/%devs_%dstns/reformatted/",min_evs,min_stns))

## model names:
model_name_list <- c('BSSA14rock_PGA_res','BSSA14_PGA_res','ASK14rock_PGA_res','ASK14_PGA_res')

## Project directory:
project_directory <- sprintf('/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/%devs_%dstns/', min_evs, min_stns)

## Data file:
data_path <- sprintf('/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals_culled-%devs-%dstns.csv', min_evs, min_stns)

##############################################################
############## Functions needed for mixed effects ##############

# Method to get fixed effects

get_fixed <- function(model, ...){ UseMethod("get_fixed") }

get_fixed.merMod <- function(model, ...){
  require(lme4)
  # get a data.frame of the fixed effects
  coef.tbl <- {
    modsum <- summary(model) #Can also use 'fixef' but this doesn't return std errors
    coefs <- modsum[['coefficients']]
    colnames(coefs) <- c('Estimate','Std.error','t.value')
    coefs <- cbind(data.frame(fixed.effect=rownames(coefs)), coefs)
    rownames(coefs) <- NULL
    return(coefs)
  }
  return(coef.tbl)
}

####################################################################

# Method to get the site random effects

get_site <- function(model, ...){ UseMethod("get_site") }

get_site.merMod <- function(model, ...){
  require(lme4)
  # get a data.frame of the random site effects
  modran <- lme4::ranef(model, condVar=TRUE)
  # internal function to add the std errors to the output
  .add_stderr <- function(x){
    id <- rownames(x)
    bias <- x$`(Intercept)`
    vars <- as.vector(attr(x, 'postVar'))
    std.err <- sqrt(vars)
    return(data.frame(ID=id, Bias=bias, Std.error=std.err))
  }
  
  coefs <- .add_stderr(modran[['Station']])
  return(coefs)
}

####################################################################

# Method to get the event random effects

get_event <- function(model, ...){ UseMethod("get_event") }

get_event.merMod <- function(model, ...){
  require(lme4)
  # get a data.frame of the random event effects
  modran <- lme4::ranef(model, condVar=TRUE)
  # internal function to add the std errors to the output
  .add_stderr <- function(x){
    id <- rownames(x)
    bias <- x$`(Intercept)`
    vars <- as.vector(attr(x, 'postVar'))
    std.err <- sqrt(vars)
    return(data.frame(ID=id, Bias=bias, Std.error=std.err))
  }
  
  coefs <- .add_stderr(modran[['Event']])
  return(coefs)
}

##############################################################

### Run the models...

## Libraries
require(lme4)

## Read in file:
db <- read.csv(data_path,sep=",")

for (i_gmml in 1:length(model_name_list))
{  
  print(i_gmml)

  i_model_name <- model_name_list[i_gmml]
  print(i_model_name)
  
  ## make the formula string, only random effects, the event and site
  i_formula_string <- paste(i_model_name, '~ (1|Event) + (1|Station)')
  ## run the model
  i_model <- lme4::lmer(i_formula_string,data=db)
  ## print it
  print(i_model)
  
  ## Extract the bias term, the fixed coefficient 
  fixed_coefs <- get_fixed(i_model)
  ## get the filepath, write the file
  fixed_file <- paste(project_directory,"R_output/",location_name,"_fixed_",i_model_name,".txt",sep='')
  readr::write_csv(fixed_coefs, path=fixed_file)
  message("Fixed coefficients saved to: ", fixed_file)
  
  ## Extract the event coefficients:
  event_coefs <- get_event(i_model)
  ## get the filepath, write the file
  event_file <- paste(project_directory,"R_output/",location_name,"_event_",i_model_name,".txt",sep='')
  readr::write_csv(event_coefs, path=event_file)
  message("Event coefficients saved to: ", event_file)
  
  ## Extract the site coefficients:
  site_coefs <- get_site(i_model)
  ## get the filepath, write the file
  site_file <- paste(project_directory,"R_output/",location_name,"_site_",i_model_name,".txt",sep='')
  readr::write_csv(site_coefs, path=site_file)
  message("Site coefficients saved to: ", site_file)
}