library(tidyverse)

# Custom mutate_at with columns renaming
my_mutate_at = function(df, no_mutate, name_add = "_sum_3", fun, ...){
  df = df %>% 
    mutate_at(.vars = setdiff(colnames(.), no_mutate), .funs = list(fun), ...) 
  
  # https://github.com/tidyverse/dplyr/issues/2947
  # df %>% 
  #   rename_at(setdiff(colnames(.), no_mutate), .funs = funs(paste0(., name_add)))
  
  idx = which(colnames(df) %in% setdiff(colnames(df), no_mutate))
  
  colnames(df)[idx] = paste0(colnames(df)[idx], name_add)
  
  return(df)
}

# Generic function to apply my backwards functions
# Should be a bit cleaner & flexible 
get_backward_fun = function(df_infos, iter, no_mutate, name_add = "_mean_", 
                            fun = roll_mean, 
                            cols_group = c("store_nbr", "item_nbr"), ...){
  list_backward_fun = list()
  i = 1
  df_infos_grouped = df_infos %>%
    group_by_at(cols_group)
  
  for(k in iter){
    print(k)
    tmp_backward_fun <- df_infos_grouped %>%  
      my_mutate_at(no_mutate = no_mutate, name_add = paste0(name_add, k), 
                   fun = fun, n=k, fill=NA, align=c("right"), ...) %>% 
      ungroup()
    
    tmp_backward_fun = tmp_backward_fun %>% 
      select(-date) %>% 
      select(-one_of(cols_group))
    
    list_backward_fun[[i]] = tmp_backward_fun # Ugly R syntax
    i = i + 1
  }
  
  backward_fun = bind_cols(df_infos %>% 
                             select(date, item_nbr, store_nbr), # Row keys
                           list_backward_fun)
  
  return(backward_fun)
}
