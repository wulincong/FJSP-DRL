# python ./test_script/trained_model.py --test_data 40x5+mix --test_model 40x5x0+mix
# python ./test_script/trained_model.py --test_data 40x10+mix --test_model 40x10x0+mix
# python ./test_script/trained_model.py --test_data 40x15+mix --test_model 40x15x0+mix
# python ./test_script/trained_model.py --test_data 40x20+mix --test_model 40x20x0+mix
# python ./test_script/trained_model.py --test_data 40x25+mix --test_model 40x25x0+mix
# python ./test_script/trained_model.py --test_data 40x20+mix --test_model 40x20x0+mix
# python ./test_script/trained_model.py --test_data 40x10+mix --test_model 40x10x0+mix
# python ./test_script/trained_model.py --test_data 30x5+mix --test_model 30x5x0+mix
# python ./test_script/trained_model.py --test_data 30x10+mix --test_model 30x10x0+mix
# python ./test_script/trained_model.py --test_data 30x15+mix --test_model 30x15x0+mix
# python ./test_script/trained_model.py --test_data 30x20+mix --test_model 30x20x0+mix
# python ./test_script/trained_model.py --test_data 30x25+mix --test_model 30x25x0+mix
# python ./test_script/trained_model.py --test_data 30x20+mix --test_model 30x20x0+mix
# python ./test_script/trained_model.py --test_data 30x10+mix --test_model 30x10x0+mix
python ./test_script/trained_model.py --test_data 40x10+mix 40x15+mix 40x20+mix 40x25+mix 40x20+mix 40x10+mix 30x5+mix 30x10+mix 30x15+mix 30x20+mix 30x25+mix 30x20+mix 30x10+mix --test_model maml+exp18-10 --hidden_dim_actor 512 --hidden_dim_critic 512
