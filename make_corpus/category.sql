DROP PROCEDURE IF EXISTS main;

DELIMITER //

CREATE PROCEDURE main()
BEGIN

	SET @parent = '政治';
	DROP TABLE IF EXISTS wiki_temp.society; 
	CREATE TABLE wiki_temp.society(id INT PRIMARY KEY,category TEXT);
	INSERT INTO wiki_temp.society VALUE (0,@parent);

	SET @i = 0;
	SET @c = 0;
    DROP TEMPORARY TABLE IF EXISTS temp; 
	CREATE TEMPORARY TABLE temp(cat TEXT);
    
	myloop: WHILE TRUE DO
		SELECT category FROM wiki_temp.society WHERE id = @i LIMIT 1 INTO @parent;
		INSERT INTO temp SELECT page_title AS child 
			FROM page JOIN categorylinks ON categorylinks.cl_from=page.page_id 
			WHERE categorylinks.cl_type = "subcat" AND categorylinks.cl_to = @parent;
		DELETE FROM temp WHERE EXISTS(SELECT category FROM wiki_temp.society WHERE temp.cat = wiki_temp.society.category);
		SELECT COUNT(*) FROM wiki_temp.society into @c;
		SELECT COUNT(*) FROM temp into @t;
        SET @j = 0;
		WHILE @t > @j DO
			SELECT cat FROM temp LIMIT 1 INTO @ccat;
			INSERT INTO wiki_temp.society(id , category) VALUES(@c+@j,@ccat);
			DELETE FROM temp WHERE temp.cat = @ccat;
			SET @j = @j + 1;
		END WHILE;
		SELECT COUNT(*) FROM wiki_temp.society INTO @c;
		IF @c = @i THEN
			LEAVE myloop;
		END IF;
		IF @i%1000=0 AND @i > 0 THEN
			SELECT @i,@c,@parent;
		END IF;
		SET @i = @i + 1;
	END WHILE myloop;

END //
CALL main();

	